"""
gbma.py
=======
GBMA (Gaussian Beam Method of Analysis) —— 馈源分解 → 逐束反射 → 远场相干合成

链路
----
1) **馈源分解** (`FeedGBExpansion`)
   用一小族旋转对称高斯波束覆盖馈源照射的立体角(六边形格点, 共 1+3K(K+1) 束),
   每束的束腰放在馈源相位中心 W, 腰半径由格点角距与重叠比 overlap 决定:
        theta_gb = (theta_cover / K) / overlap ,  w0 = 2/(k*theta_gb),  b = k*w0^2/2
   展开系数 c_n 由**反射面上的点匹配**最小二乘确定(Chou 的 GB shooting 思路):
        Σ_n c_n Ψ_n(O_m) ≈ H_feed(O_m),   O_m = 第 m 束轴与反射面的交点
   其中 Ψ_n 是"轴上幅度归一"的 GB, H_feed 用与 PO 通路完全相同的馈源模型:
        H_feed(r) = F(theta) * exp(-jkR)/R * h_hat,   h_hat = normalize(k_hat × x')
   (F(theta) 就是 feed_antenna.FeedAntenna.far_field_amplitude)

2) **逐束反射** (`ChouPathakSingle.Ps`, 来自 gb_single.py)
   每束在自己的反射点 O_n 上建局部模型(局部抛物面 + 投影边缘椭圆拟合),
   得到 P̄^s_n(S_n) —— 论文式 (74) 的 T·P̄_r + P̄_d(用数值稳定的等价形式)。

3) **合成** (`gbma_far_field`)
        Ω(S) = Σ_n Rot_n · P̄^s_n(Rot_n^T S) · exp(+j k S·(O_n - O_0))
   Rot_n   : 第 n 束局部基 → 全局基 (x_hat, y_hat, z_hat 的三行矩阵)
   exp(...) : 各束的远场相位参考点从 O_n 平移到公共原点 O_0
   (每束的入射相位 e^{-jkR_n} 已经含在系数 c_n 里, 所以相干叠加是正确的)

约定
----
* 长度单位: 米(c.py 里 freq/F/D 都是米)。gb_single 内部也直接用米的 k, 单位自洽。
* 阻抗: c.py 的 PO 通路(`compute_incident_H` + `POIntegrator`)里没有 Z0,
  为了让两条曲线能直接叠画, GBMA 输出同样**除以 Z0**(见 `_IMPEDANCE_SCALE`)。
* 束数: n_beams 会被吸附到最近的六边形格点数 1,7,19,37,61,91,127,169,217,271,...
"""

from __future__ import annotations

import time

import numpy as np

import gb_single as gs

_PI = np.pi
# c.py 的 PO 通路里没有自由空间波阻抗, 这里同样把它除掉, 使两条曲线同一标定
_IMPEDANCE_SCALE = 1.0 / gs.Z0


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _orthobasis(ax, ref=None):
    """给定单位轴 ax, 返回与它正交的两个单位矢 (e1, e2)"""
    ax = _normalize(ax)
    if ref is None:
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, ax)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
    e1 = np.asarray(ref, dtype=float) - np.dot(ref, ax) * ax
    if np.linalg.norm(e1) < 1e-12:
        ref = np.array([1.0, 0.0, 0.0])
        e1 = np.asarray(ref, dtype=float) - np.dot(ref, ax) * ax
    e1 = _normalize(e1)
    e2 = np.cross(ax, e1)
    return e1, e2


def _cone_dir(ax, theta, phi, e1, e2):
    """绕极轴 ax 的球坐标方向"""
    return (np.cos(theta) * ax
            + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2))


# --------------------------------------------------------------------------- #
# 反射面
# --------------------------------------------------------------------------- #
class ParabolicReflector:
    """抛物面 z = (x^2+y^2)/(4F) 上的口径圆盘:
    圆心 (offset_x, offset_y, ·) 在口径面内, 半径 a = D/2。
    口径面内的圆盘就是"被照射区域", 它的边界即边缘 edge。
    """

    def __init__(self, F, D, offset_x=0.0, offset_y=0.0):
        self.F = float(F)
        self.D = float(D)
        self.a = 0.5 * self.D
        self.ox = float(offset_x)
        self.oy = float(offset_y)
        # gb_single 只用到 .F / .a, 这里给个占位 Paraboloid 供 LocalModelFit 读取
        self.parab = gs.Paraboloid(self.F, self.a)

    def z(self, x, y):
        return (x * x + y * y) / (4.0 * self.F)

    def rim(self, n=1440):
        """口径边缘(反射面坐标系)"""
        t = np.linspace(0.0, 2.0 * _PI, n, endpoint=False)
        x = self.ox + self.a * np.cos(t)
        y = self.oy + self.a * np.sin(t)
        return np.stack([x, y, self.z(x, y)], axis=-1)

    def inside(self, P):
        return ((P[0] - self.ox) ** 2 + (P[1] - self.oy) ** 2) <= self.a ** 2 + 1e-12

    def center(self):
        return np.array([self.ox, self.oy, self.z(self.ox, self.oy)])

    def focus(self):
        return np.array([0.0, 0.0, self.F])

    def hit_ray(self, W, d):
        """射线 W + t*d (t>0) 与抛物面的交点, 返回最近交点或 None"""
        W = np.asarray(W, dtype=float)
        d = _normalize(d)
        A = (d[0] ** 2 + d[1] ** 2) / (4.0 * self.F)
        B = 2.0 * (W[0] * d[0] + W[1] * d[1]) / (4.0 * self.F) - d[2]
        C = (W[0] ** 2 + W[1] ** 2) / (4.0 * self.F) - W[2]
        roots = []
        if abs(A) < 1e-14:
            if abs(B) > 1e-14:
                roots = [-C / B]
        else:
            disc = B * B - 4.0 * A * C
            if disc >= 0.0:
                sq = np.sqrt(disc)
                roots = [(-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)]
        roots = [t for t in roots if t > 1e-9]
        if not roots:
            return None
        return W + min(roots) * d

    def theta_extent(self, W, ax, n=720):
        """从 W 沿 ax 看过去, 口径边缘的最大张角(用于确定格点覆盖范围)"""
        P = self.rim(n) - np.asarray(W, dtype=float)
        R = np.linalg.norm(P, axis=1)
        cosang = np.clip((P @ _normalize(ax)) / R, -1.0, 1.0)
        return float(np.arccos(cosang.min()))


# --------------------------------------------------------------------------- #
# 馈源 GB 展开
# --------------------------------------------------------------------------- #
class GBBeam:
    """一束 GB 的完整描述(反射点、局部模型、闭式解器、展开系数)"""

    __slots__ = ("O", "L", "dir", "theta", "lm", "beam", "solver", "coef", "inside",
                 "_analytic", "_goedge", "goedge_takenaka", "goedge_crescent")

    def __init__(self, O, L, dir, theta, lm, beam, inside,
                 edge_crescent=False, crescent_order=1,
                 goedge_takenaka=True, goedge_crescent=True):
        self.O, self.L, self.dir, self.theta = O, L, dir, theta
        self.lm, self.beam, self.inside = lm, beam, inside
        self.solver = gs.ChouPathakSingle(lm, beam, edge_crescent=edge_crescent,
                                          crescent_order=crescent_order)
        self._analytic = None
        self._goedge = None
        # goedge 路径的两个选项（REPORT A.28.6/A.28.7），只影响 form="goedge"
        self.goedge_takenaka = bool(goedge_takenaka)
        self.goedge_crescent = bool(goedge_crescent)
        self.coef = 1.0 + 0.0j

    @property
    def analytic(self):
        """纯解析路径的解器 (REPORT A.27), 首次用到时才建"""
        if self._analytic is None:
            self._analytic = gs.ChouPathakAnalytic(self.lm, self.beam, k=self.solver.k)
        return self._analytic

    @property
    def goedge(self):
        """GO 反射 + PO 边缘衍射 (REPORT A.28), 首次用到时才建"""
        if self._goedge is None:
            import gb_go_edge as _ge
            self._goedge = _ge.GoEdgeBeam(self.lm, self.beam, k=self.solver.k,
                                          takenaka=self.goedge_takenaka,
                                          crescent=self.goedge_crescent)
        return self._goedge


class FeedGBExpansion:
    """把馈源的照射场展开成一族高斯波束(见模块 docstring)"""

    def __init__(self, feed, reflector, n_beams=91, overlap=1.2, theta_fill=1.0,
                 reg=1e-8, verbose=True, rim_n=1440, rim_fit_n=240, sing_vals=False,
                 edge_crescent=False, crescent_order=1,
                 goedge_takenaka=True, goedge_crescent=True):
        self.feed = feed
        self.k = float(feed.k)
        self.refl = reflector
        self.overlap = float(overlap)
        self.theta_fill = float(theta_fill)
        self.reg = float(reg)
        self.verbose = verbose
        # edge_crescent: 给每束的截断因子加"边缘曲率(月牙)"修正 (REPORT A.24/A.25)
        self.edge_crescent = bool(edge_crescent)
        self.crescent_order = int(crescent_order)
        # goedge 路径的两个选项, 只影响 form="goedge" (REPORT A.28.6/A.28.7)
        self.goedge_takenaka = bool(goedge_takenaka)
        self.goedge_crescent = bool(goedge_crescent)
        # rim_n     : 求 theta_extent / 口径内判定用的边缘采样数（要足够密）
        # rim_fit_n : LocalModelFit 拟合椭圆用的边缘点数（抽稀可省 ~65 ms/91 束）
        # sing_vals : 是否算点匹配矩阵的奇异值（纯诊断，273×91 的 SVD 约 15 ms）
        self.rim_n = int(rim_n)
        self.rim_fit_n = int(rim_fit_n)
        self.want_sing_vals = bool(sing_vals)

        self.W = reflector.focus()
        self.ax = _normalize(reflector.center() - self.W)      # 馈源轴(指向反射面)
        self.theta_rim = reflector.theta_extent(self.W, self.ax)
        # 最外环正好落在口径边缘上(theta_fill=1): 这样每束的反射点都在口径内,
        # 边缘效应由 (74) 的 P_d 负责; 想让格点略微超出可把 theta_fill 调大
        self.theta_cover = self.theta_fill * self.theta_rim
        self.K = self._rings_for(int(n_beams))
        self.n_beams = 1 + 3 * self.K * (self.K + 1)
        self.dtheta = self.theta_cover / max(self.K, 1)
        self.theta_gb = self.dtheta / self.overlap
        self.w0 = 2.0 / (self.k * max(self.theta_gb, 1e-6))
        self.b = 0.5 * self.k * self.w0 ** 2

        # 参考极化(与 compute_incident_H 一致): x' = x_global 去掉 ax 分量
        xg = np.array([1.0, 0.0, 0.0])
        xp = xg - np.dot(xg, self.ax) * self.ax
        if np.linalg.norm(xp) < 1e-9:
            xg = np.array([0.0, 1.0, 0.0])
            xp = xg - np.dot(xg, self.ax) * self.ax
        self.x_prime = _normalize(xp)

        self.rim = reflector.rim(self.rim_n)
        # 拟合用的边缘子集（等间隔抽稀；椭圆拟合对点数不敏感）
        step = max(1, self.rim_n // max(self.rim_fit_n, 8))
        self.rim_fit = self.rim[::step]
        self.beams = []
        self.residual = None

    # ---- 六边形格点数 ----
    @staticmethod
    def _rings_for(n_beams):
        best, bestd = 1, None
        for K in range(1, 40):
            tot = 1 + 3 * K * (K + 1)
            d = abs(tot - n_beams)
            if bestd is None or d < bestd:
                best, bestd = K, d
        return best

    # ---- 格点 ----
    def lattice(self):
        e1, e2 = _orthobasis(self.ax)
        dirs, ths = [self.ax], [0.0]
        for i in range(1, self.K + 1):
            ni = 6 * i
            for j in range(ni):
                phi = 2.0 * _PI * j / ni
                dirs.append(_cone_dir(self.ax, i * self.dtheta, phi, e1, e2))
                ths.append(i * self.dtheta)
        return np.asarray(dirs), np.asarray(ths)

    # ---- 馈源入射场(与 compute_incident_H 同模型) ----
    def feed_H(self, P):
        """H_feed(r) = F(theta) e^{-jkR}/R h_hat (可对 (...,3) 批量)"""
        P = np.asarray(P, dtype=float)
        V = P - self.W
        R = np.linalg.norm(V, axis=-1)
        khat = V / R[..., None]
        cosang = np.clip(khat @ self.ax, -1.0, 1.0)
        theta = np.arccos(cosang)
        amp = self.feed.far_field_amplitude(theta) / R
        ph = np.exp(-1j * self.k * R) * amp
        h = np.cross(khat, self.x_prime)
        hn = np.linalg.norm(h, axis=-1, keepdims=True)
        h = np.where(hn > 1e-12, h / np.where(hn > 1e-12, hn, 1.0), 0.0)
        return ph[..., None] * h

    def _h_hat(self, d):
        """单方向的极化单位矢"""
        h = np.cross(_normalize(d), self.x_prime)
        n = np.linalg.norm(h)
        return h / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])

    # ---- 分解主体 ----
    def build(self):
        dirs, ths = self.lattice()
        beams = []
        for d, th in zip(dirs, ths):
            O = self.refl.hit_ray(self.W, d)
            if O is None:
                continue
            lm = gs.LocalModelFit(self.refl.parab, O, self.rim_fit, verbose=False)
            R = float(np.linalg.norm(O - self.W))
            to_l = np.stack([lm.xhat, lm.yhat, lm.zhat])       # 行 = 局部基(全局分量)
            zi_l = to_l @ _normalize(d)
            pol_l = to_l @ self._h_hat(d)
            beam = gs.GaussianBeam(self.b, R, zi_l, pol_ref=pol_l, k=self.k)
            beams.append(GBBeam(O=O, L=R, dir=_normalize(d), theta=float(th), lm=lm,
                                beam=beam, inside=bool(self.refl.inside(O)),
                                edge_crescent=self.edge_crescent,
                                crescent_order=self.crescent_order,
                                goedge_takenaka=self.goedge_takenaka,
                                goedge_crescent=self.goedge_crescent))
        if not beams:
            raise RuntimeError("GBMA: 没有任何束轴与反射面相交")
        self.beams = beams
        # 诊断: 每束反射点处的入射角(相对局部法向)。论文假设"不接近掠入射";
        # 实测 >~40deg 时闭合解精度明显下滑(见 REPORT.md 附录 A.6)
        inc = []
        for bb in beams:
            dv = bb.O - self.W
            dv = dv / np.linalg.norm(dv)
            inc.append(np.degrees(np.arccos(min(1.0, abs(dv @ bb.lm.zhat)))))
        self.incidence_deg = np.asarray(inc, dtype=float)
        self.max_incidence_deg = float(np.max(self.incidence_deg))
        self._solve_coefficients()
        if self.verbose:
            n_out = sum(0 if b.inside else 1 for b in beams)
            print(f"  [GB 分解] 束数={len(beams)} (环数 K={self.K}), "
                  f"theta_cover={np.degrees(self.theta_cover):.2f}°, "
                  f"Δθ={np.degrees(self.dtheta):.2f}°, theta_gb={np.degrees(self.theta_gb):.2f}°")
            print(f"             w0={self.w0*1e3:.2f} mm, b={self.b*1e3:.2f} mm, "
                  f"反射点在口径内 {len(beams)-n_out}/{len(beams)}, "
                  f"点匹配残差={self.residual:.3e}")
            print(f"             入射角: 最大 {self.max_incidence_deg:.1f}deg, "
                  f"中位 {np.median(self.incidence_deg):.1f}deg")
            if self.max_incidence_deg > 40.0:
                print(f"             警告: 最大入射角 {self.max_incidence_deg:.1f}deg 偏大; "
                      f"chou1997 的闭式解要求不接近掠入射, 大偏置几何精度会下降")
        return self

    def _solve_coefficients(self):
        """点匹配: Σ_n c_n Ψ_n(O_m) = H_feed(O_m)  (Ψ 为单位轴上幅度的 GB)

        矩阵**按列**装配: 第 n 列是"第 n 束在全部 N 个匹配点上的场"。GaussianBeam.H
        与 LocalModel.to_local 都是向量化的, 因此把整批点一次传入, 每列只需 1 次调用;
        原来是按点先行的双层循环, 要 N² 次调用 (N=91 时 8281 次, 实测占分解耗时的
        ~88%). 两种写法得到**逐位一致**的矩阵 (相对差 <1e-15)。
        """
        N = len(self.beams)
        Rls = [np.stack([b.lm.xhat, b.lm.yhat, b.lm.zhat]) for b in self.beams]
        Oall = np.array([b.O for b in self.beams])                  # (N,3)
        A = np.zeros((3 * N, N), dtype=complex)
        for n, bn in enumerate(self.beams):
            loc = bn.lm.to_local(Oall)                              # (N,3) 本束局部系
            psi = bn.beam.H(loc) @ Rls[n]                           # (N,3) 全局分量
            A[:, n] = psi.reshape(-1)                               # 行序 = 点序×分量
        # 馈源场同样一次算完 (feed_H 支持批量)
        rhs = self.feed_H(Oall).reshape(-1)                          # (3N,)
        A_unit, rhs_unit = A.copy(), rhs.copy()
        # 轻微 Tikhonov 正则, 防止格点过密时系数爆炸
        if self.reg > 0:
            scale = np.linalg.norm(A) / np.sqrt(max(A.size, 1))
            A = np.vstack([A, np.sqrt(self.reg) * scale * np.eye(N, dtype=complex)])
            rhs = np.concatenate([rhs, np.zeros(N, dtype=complex)])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        self.sol = sol
        self.sing_vals = (np.linalg.svd(A_unit, compute_uv=False)
                          if self.want_sing_vals else None)
        for b, c in zip(self.beams, sol):
            b.coef = complex(c)
            b.beam.H0 = c * b.beam.xi_dir.astype(complex)     # 轴上幅度 × 展开系数
        # 残差(用未加正则、未乘系数的那套方程)
        self.residual = float(np.linalg.norm(A_unit @ sol - rhs_unit)
                              / max(np.linalg.norm(rhs_unit), 1e-30))

    def check_expansion(self, n_samples=400, seed=0, verbose=True):
        """直接检验"馈源分解": 在反射面上抽样比较 Σ_n c_n Ψ_n(r) 与 H_feed(r)"""
        rng = np.random.default_rng(seed)
        rr = self.refl.a * np.sqrt(rng.random(n_samples))
        pp = 2.0 * np.pi * rng.random(n_samples)
        x = self.refl.ox + rr * np.cos(pp)
        y = self.refl.oy + rr * np.sin(pp)
        P = np.stack([x, y, self.refl.z(x, y)], axis=-1)
        target = self.feed_H(P)
        approx = np.zeros_like(target)
        for b in self.beams:
            psi = b.beam.H(b.lm.to_local(P))
            Rl = np.stack([b.lm.xhat, b.lm.yhat, b.lm.zhat])
            approx += psi @ Rl
        num = np.linalg.norm(approx - target, axis=-1)
        den = np.linalg.norm(target, axis=-1)
        rel = num / np.maximum(den, 1e-30)
        amp = np.linalg.norm(target, axis=-1)
        w = amp ** 2 / np.sum(amp ** 2)
        rel_w = float(np.sqrt(np.sum(w * rel ** 2)))
        if verbose:
            print(f"  [分解检验] 面上相对误差: 光强加权 RMS={rel_w:.3e}, "
                  f"中位数={np.median(rel):.3e}, 95%分位={np.percentile(rel, 95):.3e}")
        return {"rel_rms_weighted": rel_w, "rel_median": float(np.median(rel)),
                "rel_p95": float(np.percentile(rel, 95))}


# --------------------------------------------------------------------------- #
# 远场匹配版分解 (chou2001 §II) —— 参考 gb_farfield_decomp.py
# --------------------------------------------------------------------------- #
class FarFieldFeedGBExpansion(FeedGBExpansion):
    """在**馈源远场方向图**上做点匹配的 GB 分解（chou2001 §II，论文推荐做法）。

    与近场版 `FeedGBExpansion` 的差别只有两处：

    * **参数怎么选**：(13)(15)(19) 那条链——(19) 窗口里按 `w_rel` 选光斑 w →
      (15) 小根解出共焦参数 b → (13) 得角距 Δθ → 六边形格点覆盖锥，**N 是产物**。
      （近场版反过来：给定束数 N → Δθ → θ_gb → w0 → b。）
    * **系数怎么定**：在"加密一倍的远场测试格点"上解
      `Σ_n A_n Ψ_n(∠(ŝ, ŝ_n)) = F_feed(θ)`，再换算到反射点处的入射场幅度
      `H0_n = A_n·e^{−jkL_n}/(L_n + jb)`（换算的推导见 REPORT A.18；
      少了 `jb` 会差 ×1.4/+45°，是这一步最容易写错的地方）。

    基函数两支：
      `basis="paper"` : Ψ = exp[−(kb/2)sin²α]·(1+cosα)/2   —— gb_farfield_decomp 的写法
      `basis="beam"`  : Ψ = exp[−(tanα/θ_gb)²]/cosα        —— 与本工程 GaussianBeam 的
                        严格远区渐近极限一致（A.18 数值验证到 1e-6）

    用法与 `FeedGBExpansion` 完全一致（`build()` 之后直接喂 `gbma_far_field`）。
    """

    def __init__(self, feed, reflector, n_beams=None, w_rel=0.40, cone_margin=1.10,
                 reg=1e-8, verbose=True, rim_n=1440, rim_fit_n=240, sing_vals=False,
                 basis="paper", tilt=True, edge_crescent=False, crescent_order=1,
                 goedge_takenaka=True, goedge_crescent=True):
        if basis not in ("paper", "beam"):
            raise ValueError(f"basis 只能是 paper/beam，收到 {basis!r}")
        # 先按父类把几何/极化/rim 等公共量装好（n_beams 只影响被覆盖的格点参数）
        super().__init__(feed, reflector, n_beams=(n_beams or 91), overlap=1.2,
                         theta_fill=1.0, reg=reg, verbose=verbose,
                         rim_n=rim_n, rim_fit_n=rim_fit_n, sing_vals=sing_vals,
                         edge_crescent=edge_crescent, crescent_order=crescent_order,
                         goedge_takenaka=goedge_takenaka,
                         goedge_crescent=goedge_crescent)
        import gb_farfield_decomp as gfd

        self.gfd = gfd
        self.basis = basis
        self.tilt = bool(tilt)
        self.w_rel = float(w_rel)
        self.cone_margin = float(cone_margin)
        self.n_beams_min = None if n_beams is None else int(n_beams)

        refl_ff = gfd.Reflector(F=reflector.F, D=reflector.D,
                                offset_x=reflector.ox, offset_y=reflector.oy,
                                cone_margin=self.cone_margin, n_rim=self.rim_n)
        feed_ff = gfd.CallableFeed(lambda th: feed.far_field_amplitude(th))
        self.dec = gfd.decompose(refl_ff, feed_ff, freq=feed.freq, w_rel=self.w_rel,
                                 tilt=self.tilt, n_beams=self.n_beams_min,
                                 verbose=False)

        # 用 (13)(15)(19) 的结果覆盖父类按"束数→角距"反推的那套参数
        self.b = float(self.dec.b)
        self.w0 = float(np.sqrt(2.0 * self.b / self.k))
        self.dtheta = float(self.dec.dtheta)
        self.theta_gb = float(2.0 / (self.k * self.w0))
        self.theta_cover = float(self.dec.cone_half)
        self.n_beams = int(self.dec.N)
        self.K = 0                       # 远场版不是"环数"参数化
        self.refl_ff = refl_ff
        if verbose:
            d = self.dec.info
            print(f"  [GB 分解(farfield)] N={self._n_beams_str()} "
                  f"(N(13)={d['n_13']}), cone={np.degrees(self.theta_cover):.2f}°, "
                  f"Δθ={np.degrees(self.dtheta):.2f}°, θ_gb={np.degrees(self.theta_gb):.2f}°")
            print(f"             w={d['w']*1e3:.1f} mm (窗口 [{d['w_lo']*1e3:.1f}, "
                  f"{d['w_hi']*1e3:.1f}] mm, w_rel={self.w_rel:g}), b={self.b*1e3:.1f} mm, "
                  f"w0={self.w0*1e3:.1f} mm")
            print(f"             (16)={d['eq16']} (18)={d['eq18']} (19)={d['eq19']}; "
                  f"LSQ rel={d['lsq_rel']:.2e}, 基函数={self.basis}")

    def _n_beams_str(self):
        return f"{self.n_beams}" + ("" if not self.dec.info.get("n_override")
                                    else f"(N/N13={self.dec.info['n_ratio']:.1f})")

    # ---- 格点：馈源远场测试出来的方向 ----
    def lattice(self):
        dirs = np.asarray(self.dec.dirs, dtype=float)
        ths = np.arccos(np.clip(dirs @ self.ax, -1.0, 1.0))
        return dirs, ths

    # ---- 基函数 ----
    def _basis_ang(self, cosg):
        """Ψ(γ)，γ = arccos(cosg)；轴上 Ψ(0)=1。"""
        if self.basis == "paper":
            return (np.exp(-(self.k * self.b / 2.0) * (1.0 - cosg ** 2))
                    * ((1.0 + cosg) / 2.0 if self.tilt else 1.0))
        gam = np.arccos(np.clip(cosg, -1.0, 1.0))
        g = np.minimum(gam, 1.2)
        return np.where(gam < 1.2,
                        np.exp(-(np.tan(g) / self.theta_gb) ** 2) / np.cos(g), 0.0)

    # ---- 系数：远场 LSQ + A.18 的换算 ----
    def _solve_coefficients(self):
        N = len(self.beams)
        L = np.array([b.L for b in self.beams])
        dirs = np.array([b.dir for b in self.beams])
        # 测试格点：与格点同一套六边形排布，但角距减半（原文允许的"减小 Δθ"）
        test_z = self.gfd.hex_directions(self.theta_cover, self.dtheta / 2.0)
        S = self.gfd.rotate_to_axis(test_z, self.ax)
        A = self._basis_ang(np.clip(S @ dirs.T, -1.0, 1.0))          # (M,N)
        theta_t = np.arccos(np.clip(S @ self.ax, -1.0, 1.0))
        y = np.asarray(self.feed.far_field_amplitude(theta_t), dtype=float)
        A_unit, y_unit = A.copy(), y.copy()
        if self.reg > 0:                    # 轻微 Tikhonov（远场版原文没要求，留着防病态）
            sc = np.linalg.norm(A) / np.sqrt(max(A.size, 1))
            A = np.vstack([A, np.sqrt(self.reg) * sc * np.eye(N)])
            y = np.concatenate([y, np.zeros(N)])
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.sol = sol
        self.sing_vals = (np.linalg.svd(A_unit, compute_uv=False)
                          if self.want_sing_vals else None)
        # 远区系数 A_n -> 反射点处入射场幅度 H0_n（A.18 的 (★)）
        h0 = sol * np.exp(-1j * self.k * L) / (L + 1j * self.b)
        for b, c in zip(self.beams, h0):
            b.coef = complex(c)
            b.beam.H0 = c * b.beam.xi_dir.astype(complex)
        self.residual = float(np.linalg.norm(A_unit @ sol - y_unit)
                              / max(np.linalg.norm(y_unit), 1e-30))


# --------------------------------------------------------------------------- #
# 远场合成
# --------------------------------------------------------------------------- #
def directions(theta_deg, phi_deg):
    """观察方向(全局基), theta 从 +z 轴起算"""
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    ph = np.deg2rad(np.asarray(phi_deg, dtype=float))
    return np.stack([np.sin(th) * np.cos(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(th)], axis=-1)


def _go_fast(expansion, k=None):
    """取得（并缓存）该展开的纯 GO 精简求值器 —— 见 gb_go_edge.GoReflectionFast"""
    import gb_go_edge as _ge
    obj = getattr(expansion, "_go_fast", None)
    if obj is None or (k is not None and obj.k != float(k)):
        obj = _ge.GoReflectionFast(expansion, k=k)
        expansion._go_fast = obj
    return obj


def _goedge_fast(expansion, k=None):
    """取得（并缓存）goedge 的扁平化加速版 —— 见 gb_go_edge.GoEdgeFast (REPORT A.32)"""
    import gb_go_edge as _ge
    obj = getattr(expansion, "_goedge_fast", None)
    if obj is None or (k is not None and obj.k != float(k)):
        obj = _ge.GoEdgeFast(expansion, k=k,
                             crescent=bool(getattr(expansion, "goedge_crescent", True)))
        expansion._goedge_fast = obj
    return obj


def _literal_fast(expansion, k=None):
    """取得（并缓存）literal 的扁平化加速版 —— 见 gb_literal_fast.LiteralFast (REPORT A.38)

    与逐束 `ChouPathakSingle.Ps` 数值等价（实测相对差 8e-16），但整份展开一次算完。
    只支持生产默认那条纯闭式路径（edge_q="frozen"、无月牙、quartic_pure）。
    """
    import gb_literal_fast as _lf
    obj = getattr(expansion, "_literal_fast", None)
    if obj is None or (k is not None and obj.k != float(k)):
        obj = _lf.LiteralFast(expansion, k=k)
        expansion._literal_fast = obj
    return obj


def gbma_far_field(expansion, S, origin=None, stable=True, per_beam=False, k=None,
                   form=None, phi1se_mode="newton"):
    """合成远场 Ω(S) = Σ_n Rot·P̄^s_n(Rot^T S)·exp(+jk S·(O_n-O_0))

    phi1se_mode: 边缘点 φ₁ₛ 的解法, 透传给 ChouPathakSingle.Ps
        'newton'(默认) / 'cubic' / 'cubic_pure' / 'quartic_pure' / 'closed'
        见 REPORT A.21。

    返回 (Omega (M,3), info dict)
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    nS = len(S)
    if origin is None:
        origin = np.zeros(3)
    if k is None:
        k = expansion.beams[0].solver.k
    if form is None:
        form = "stable" if stable else "literal"
    # ---- go2 / goedge2 / literal2: 扁平化的精简快速通道（整份展开一次算完）----
    if form in ("go2", "goedge2", "literal2"):
        t0 = time.perf_counter()
        if form == "go2":
            Of = _go_fast(expansion, k).far_field(S, origin)
        elif form == "goedge2":
            Of = _goedge_fast(expansion, k).far_field(S, origin)
        else:
            Of = _literal_fast(expansion, k).far_field(S, origin)
        info = {"n_beams": len(expansion.beams), "n_clip": 0, "n_transition": 0,
                "form": _FORM_NAME[form], "form_key": form,
                "phi1se_mode": phi1se_mode, "t_sweep": time.perf_counter() - t0}
        if per_beam:
            raise ValueError(f"form={form!r} 的精简通道不支持 per_beam")
        return Of * _IMPEDANCE_SCALE, info
    _KEY = {"stable": "Ps_stable", "literal": "Ps", "auto": "Ps_auto",
            "crescent": "Ps_crescent", "analytic": "Ps_analytic",
            "goedge": "Ps_analytic", "go": "Pr", "go2": "Pr", "goedge2": "Ps_analytic"}
    if form not in _KEY:
        raise ValueError(f"form 只能是 {list(_KEY)}, 收到 {form!r}")
    out = np.zeros((nS, 3), dtype=complex)
    n_clip = 0
    n_tr = 0
    contrib = np.zeros((nS, len(expansion.beams)), dtype=complex) if per_beam else None
    for j, b in enumerate(expansion.beams):
        Rl = np.stack([b.lm.xhat, b.lm.yhat, b.lm.zhat])       # 行 = 局部基
        Sl = S @ Rl.T                                          # 全局 -> 局部 分量
        if form == "analytic":
            d = b.analytic.Ps(Sl)                     # 纯解析路径 (无复根求解)
        elif form == "goedge":
            d = b.goedge.Ps(Sl)                       # GO 反射 + PO 边缘衍射
        elif form == "go2":
            # 单形式快速通道: 只算 GO 反射本体, 不碰边缘项
            term = (b.goedge.Pr_only(Sl) @ Rl) * np.exp(
                1j * k * (S @ (b.O - origin)))[:, None]
            out += term
            continue
        elif form == "go":
            # 单形式快速通道: 只算 P_r, 不碰 (B7)/端点项
            term = (b.solver.Pr_only(Sl) @ Rl) * np.exp(
                1j * k * (S @ (b.O - origin)))[:, None]
            out += term
            continue
        else:
            d = b.solver.Ps(Sl, phi1se_mode=phi1se_mode)
        P_local = np.asarray(d[_KEY[form]])
        P_glob = P_local @ Rl                                  # 局部 -> 全局
        piston = np.exp(1j * k * (S @ (b.O - origin)))
        term = P_glob * piston[:, None]
        out += term
        n_clip += int(np.asarray(d.get("n_clip", 0)))
        n_tr += int(np.sum(np.abs(np.asarray(d["Sa"])) < 1.0))
        if per_beam:
            contrib[:, j] = np.sum(np.abs(term) ** 2, axis=-1)
    _NAME = {"stable": "stable: Pr*[1+erf(Sa)]/2",
             "literal": "literal: T*Pr+Pd",
             "auto": "auto: |Sa|>1.5 用字面式, <0.7 用稳定式(平滑过渡)",
             "crescent": "crescent(嫁接式, 仅对照): Pr*([1+erf(Sa)]/2 - S_ratio)",
             "analytic": "analytic(纯解析, 无复根): P_r*([1+erf(√c_eff d_r - jk b_eff/2√c_eff)]/2 - S_ratio) [A.27]",
             "goedge": "goedge(GO 反射+PO 边缘): F0·解析远场·([1+erf(Z)]/2 - S_ratio) [A.28]",
             "go": "go(只反射, 无边缘): P_r —— 论文 L750-753 的'边缘不存在时的反射场'",
             "go2": "go2(只反射, 无边缘): goedge 的 GO 反射本体 = F0·解析远场（Rieckmann 的几何波）"}
    info = {"n_beams": len(expansion.beams), "n_clip": n_clip, "n_transition": n_tr,
            "form": _NAME[form], "form_key": form, "phi1se_mode": phi1se_mode}
    if per_beam:
        info["per_beam"] = contrib
    return out * _IMPEDANCE_SCALE, info


# 各 form 对应的数据键（供多次/多形式合成共用）
_FORM_KEY = {"stable": "Ps_stable", "literal": "Ps", "auto": "Ps_auto",
             "crescent": "Ps_crescent", "go": "Pr", "go2": "Pr"}
_FORM_NAME = {"stable": "stable: Pr*[1+erf(Sa)]/2",
              "literal": "literal: T*Pr+Pd",
              "auto": "auto: |Sa|>1.5 用字面式, <0.7 用稳定式(平滑过渡)",
              "crescent": "crescent(嫁接式, 仅对照): Pr*([1+erf(Sa)]/2 - S_ratio)",
              "analytic": "analytic(纯解析, 无复根): P_r*([1+erf(√c_eff d_r - jk b_eff/2√c_eff)]/2 - S_ratio) [A.27]",
              "goedge": "goedge(GO 反射+PO 边缘): F0·解析远场·([1+erf(Z)]/2 - S_ratio) [A.28]",
              "go": "go(只反射, 无边缘): P_r —— 论文 L750-753 的'边缘不存在时的反射场'",
              "go2": "go2(只反射, 无边缘): goedge 的 GO 反射本体 = F0·解析远场（Rieckmann 的几何波）",
              "goedge2": "goedge2(扁平化 goedge): F0·解析远场·[E(Z)−(sgn/2)erfc(sgnZ) − S_ratio] [A.32]",
              "literal2": "literal2(扁平化 literal): 与 literal 逐元素同一算式, 整份展开一次算完 [A.38]"}


def gbma_far_field_multi(expansion, S, forms, origin=None, k=None,
                         phi1se_mode="newton", per_beam=False):
    """**一次遍历同时累积多个合成式**的远场 —— REPORT A.30

    动机：`ChouPathakSingle.Ps` 一次就返回 Ps / Ps_stable / Ps_auto / Pr / Pd，
    所以同一份展开下多要几种曲线，本来只是"多读几个键 + 多累加一次"。
    原先 c.py 对每个额外 form 重新扫一遍 91 束（重算 Ps），白花 2~3 倍时间。

    按**解器**分组，每束每组只调用一次：
      solver  : 'stable' / 'literal' / 'auto' / 'crescent' / 'go'
                （若只请求 'go'，走 `Pr_only` 快速通道，不算 (B7)/端点项）
      goedge  : 'goedge' / 'go2'
                （若只请求 'go2'，走 `Pr_only`，不算边缘项）
      analytic: 'analytic'

    返回 (dict{form: Omega (M,3)}, info)
    """
    forms = [str(f).lower() for f in forms]
    _allowed = list(_FORM_KEY) + ["analytic", "goedge", "goedge2", "literal2"]
    for f in forms:
        if f not in _FORM_KEY and f not in ("analytic", "goedge", "goedge2", "literal2"):
            raise ValueError(f"form 只能是 {_allowed}，收到 {f!r}")
    S = np.atleast_2d(np.asarray(S, dtype=float))
    nS = len(S)
    if origin is None:
        origin = np.zeros(3)
    if k is None:
        k = expansion.beams[0].solver.k
    out = {f: np.zeros((nS, 3), dtype=complex) for f in set(forms)}
    contrib = ({f: np.zeros((nS, len(expansion.beams)), dtype=complex)
                for f in set(forms)} if per_beam else None)
    n_clip = 0
    n_tr = 0
    solver_forms = [f for f in forms if f in _FORM_KEY and f not in ("go2", "goedge2", "literal2")]
    goedge_forms = [f for f in forms if f == "goedge"]
    need_go2 = "go2" in forms
    need_goedge2 = "goedge2" in forms
    need_literal2 = "literal2" in forms
    need_analytic = "analytic" in forms
    t_loop0 = time.perf_counter()
    t_ps = 0.0
    t_solve_by = {}                 # 独立解器(goedge/analytic)的累计求解时间
    t_acc = {}
    if need_go2:
        _t = time.perf_counter()
        out["go2"] = _go_fast(expansion, k).far_field(S, origin)
        t_acc["go2"] = time.perf_counter() - _t              # 独立批量调用, 整段都算它
        t_ps += 0.0
    if need_goedge2:
        _t = time.perf_counter()
        out["goedge2"] = _goedge_fast(expansion, k).far_field(S, origin)
        t_acc["goedge2"] = time.perf_counter() - _t
    if need_literal2:
        _t = time.perf_counter()
        out["literal2"] = _literal_fast(expansion, k).far_field(S, origin)
        t_acc["literal2"] = time.perf_counter() - _t
    if (need_go2 or need_goedge2 or need_literal2) and per_beam:
        raise ValueError("forms 含 'go2'/'goedge2'/'literal2' 时不支持 per_beam")

    def _acc(f, P_local, Rl, piston, j):
        _t = time.perf_counter()
        term = (np.asarray(P_local) @ Rl) * piston[:, None]
        out[f] += term
        if per_beam:
            contrib[f][:, j] = np.sum(np.abs(term) ** 2, axis=-1)
        t_acc[f] = t_acc.get(f, 0.0) + (time.perf_counter() - _t)

    for j, b in enumerate(expansion.beams):
        Rl = np.stack([b.lm.xhat, b.lm.yhat, b.lm.zhat])
        Sl = S @ Rl.T
        piston = np.exp(1j * k * (S @ (b.O - origin)))
        # ---- solver 组 ----
        if solver_forms:
            _t = time.perf_counter()
            if solver_forms == ["go"]:
                P = b.solver.Pr_only(Sl)                    # 快速通道: 只算 P_r
                d = None
            else:
                d = b.solver.Ps(Sl, phi1se_mode=phi1se_mode)
            t_ps += time.perf_counter() - _t
            if d is not None:
                n_clip += int(np.asarray(d.get("n_clip", 0)))
                n_tr += int(np.sum(np.abs(np.asarray(d["Sa"])) < 1.0))
                for f in solver_forms:
                    _acc(f, d[_FORM_KEY[f]], Rl, piston, j)
            else:
                _acc("go", P, Rl, piston, j)
        # ---- goedge 组（独立解器：其求解时间要归到 goedge 这条曲线头上）----
        if goedge_forms:
            _t = time.perf_counter()
            dg = b.goedge.Ps(Sl)
            t_solve_by["goedge"] = t_solve_by.get("goedge", 0.0) + (time.perf_counter() - _t)
            t_ps += time.perf_counter() - _t
            for f in goedge_forms:
                _acc(f, dg["Ps_analytic"], Rl, piston, j)
        # ---- analytic（同理）----
        if need_analytic:
            _t = time.perf_counter()
            P = b.analytic.Ps(Sl)["Ps_analytic"]
            t_solve_by["analytic"] = t_solve_by.get("analytic", 0.0) + (time.perf_counter() - _t)
            _acc("analytic", P, Rl, piston, j)
    t_loop = time.perf_counter() - t_loop0
    for f, v in t_solve_by.items():           # 独立解器的曲线, 求解时间并入其边际成本
        t_acc[f] = t_acc.get(f, 0.0) + v
    out = {f: v * _IMPEDANCE_SCALE for f, v in out.items()}
    info = {"n_beams": len(expansion.beams), "n_clip": n_clip, "n_transition": n_tr,
            "forms": forms, "phi1se_mode": phi1se_mode,
            "form": _FORM_NAME.get(forms[0], forms[0]), "form_key": forms[0],
            # 实测：整次 sweep 里"逐束求解"占多少，剩下的是多形式累加/旋转的边际成本
            "t_sweep": t_loop, "t_per_beam_solve": t_ps,
            "t_acc_by_form": dict(t_acc),
            "t_accum_all": max(t_loop - t_ps, 0.0)}
    if per_beam:
        info["per_beam"] = contrib
    return out, info


def po_reference(expansion, S, n_rr=220, n_ph=768, k=None):
    """数值 PO 金标准(用于对拍整条 GBMA 链):

        Ω(S) = (jkZ0/4π) ∬ [S×[S×(2 n̂ × H_feed)]] e^{jk S·r'} dS'

    与 c.py 的 PO 通路同一约定(末尾除掉 Z0), 入射场用**同一个** feed 模型
    (`FeedGBExpansion.feed_H`), 因此差异只来自 GBMA 自身。
    在口径极坐标 (r ∈ [0,a] 绕口径圆心, φ 均匀) 上用 GL × 均匀求积。
    """
    feed = expansion.feed_H
    refl = expansion.refl
    k = expansion.k if k is None else float(k)
    S = np.atleast_2d(np.asarray(S, dtype=float))
    nodes, wts = np.polynomial.legendre.leggauss(n_rr)
    rr = 0.5 * refl.a * (nodes + 1.0)
    wr = 0.5 * refl.a * wts
    pp = 2.0 * np.pi * np.arange(n_ph) / n_ph
    R_, P_ = np.meshgrid(rr, pp, indexing="ij")
    X = refl.ox + R_ * np.cos(P_)
    Y = refl.oy + R_ * np.sin(P_)
    Z = refl.z(X, Y)
    Pts = np.stack([X, Y, Z], axis=-1)
    px, py = X / (2.0 * refl.F), Y / (2.0 * refl.F)
    one = np.ones_like(px)
    nv = np.stack([-px, -py, one], axis=-1)
    nrm = nv / np.linalg.norm(nv, axis=-1, keepdims=True)
    dS = np.linalg.norm(nv, axis=-1) * R_ * (wr[:, None] * (2.0 * np.pi / n_ph))
    H = feed(Pts)
    J = 2.0 * np.cross(nrm, H)                                   # (11)
    Wt = (J * dS[..., None]).reshape(-1, 3)
    Pf = Pts.reshape(-1, 3)
    out = np.zeros((len(S), 3), dtype=complex)
    step = 8
    for i0 in range(0, len(S), step):
        ph = np.exp(1j * k * (S[i0:i0 + step] @ Pf.T))
        tmp = ph @ Wt
        SxS = np.cross(S[i0:i0 + step], np.cross(S[i0:i0 + step], tmp))
        out[i0:i0 + step] = SxS
    return out * (1j * k / (4.0 * np.pi))          # 含 Z0 的因子已除掉(与 GBMA 同标定)


# --------------------------------------------------------------------------- #
# 与 c.py 对接的入口
# --------------------------------------------------------------------------- #
def compute_feed_total_power(feed):
    """馈源总辐射功率(与 c.py 同一表达式, 用于方向性归一化)"""
    x = feed.k * feed.b
    return (np.pi / (8.0 * x ** 3)) * (8.0 * x ** 2 - 4.0 * x + 1.0 - np.exp(-4.0 * x))


def compute_far_field_gbma(feed, cfg, theta_scan_deg, n_beams=None, overlap=None,
                           theta_fill=None, reg=None, verbose=True,
                           phi_deg=0.0, return_expansion=False, form=None,
                           phi1se_mode=None, decomp=None, w_rel=None, cone_margin=None,
                           ff_basis=None, edge_crescent=None, crescent_order=None,
                           goedge_takenaka=None, goedge_crescent=None,
                           extra_forms=None):
    """馈源分解 → 逐束反射 → 合成的总入口。

    参数
    ----
    feed   : feed_antenna.FeedAntenna
    cfg    : c.py 的 AntennaConfig (用到 freq/F/D/offset_x/offset_y)
    theta_scan_deg : 观察角数组(度), 从反射面 +z 轴起算
    n_beams        : 束数（近场版=吸附到最近六边形格点数；远场版=最小束数）, None -> cfg
    overlap        : 格点角距 / GB 角半径, None -> cfg.gbma_overlap   （仅近场版）
    theta_fill     : 覆盖角 / 口径张角, None -> cfg.gbma_theta_fill  （仅近场版）
    reg            : 点匹配正则, None -> cfg.gbma_reg
    phi1se_mode    : 边缘点 phi1se 解法, None -> cfg.gbma_phi1se_mode
                     ('newton'/'cubic'/'cubic_pure'/'quartic_pure'/'closed', 见 REPORT A.21)
    decomp         : 'near'（默认, 反射面上点匹配）/ 'farfield'（馈源远场点匹配, chou2001 §II）
    w_rel          : 远场版在 (19) 窗口内选 w 的位置 [0,1], None -> cfg.gbma_w_rel
    cone_margin    : 远场版覆盖锥相对边缘角的余量, None -> cfg.gbma_cone_margin
    ff_basis       : 远场版基函数 'paper'/'beam', None -> cfg.gbma_ff_basis

    返回 (E (N,3) 复数, dBi (N,), info dict)
    """
    n_beams = cfg.gbma_n_beams if n_beams is None else n_beams
    overlap = getattr(cfg, "gbma_overlap", 1.2) if overlap is None else overlap
    theta_fill = getattr(cfg, "gbma_theta_fill", 1.02) if theta_fill is None else theta_fill
    reg = getattr(cfg, "gbma_reg", 1e-8) if reg is None else reg
    if form is None:
        form = getattr(cfg, "gbma_form", "auto")
    if phi1se_mode is None:
        phi1se_mode = getattr(cfg, "gbma_phi1se_mode", "newton")
    if decomp is None:
        decomp = getattr(cfg, "gbma_decomp", "near")
    if w_rel is None:
        w_rel = getattr(cfg, "gbma_w_rel", 0.40)
    if cone_margin is None:
        cone_margin = getattr(cfg, "gbma_cone_margin", 1.10)
    if ff_basis is None:
        ff_basis = getattr(cfg, "gbma_ff_basis", "paper")
    if edge_crescent is None:
        edge_crescent = getattr(cfg, "gbma_edge_crescent", False)
    if crescent_order is None:
        crescent_order = getattr(cfg, "gbma_crescent_order", 1)
    if goedge_takenaka is None:
        goedge_takenaka = getattr(cfg, "gbma_goedge_takenaka", True)
    if goedge_crescent is None:
        goedge_crescent = getattr(cfg, "gbma_goedge_crescent", True)
    if extra_forms is None:
        extra_forms = getattr(cfg, "gbma_extra_forms", ())

    refl = ParabolicReflector(cfg.F, cfg.D, cfg.offset_x, cfg.offset_y)
    if str(decomp).lower() in ("farfield", "far", "ff"):
        exp = FarFieldFeedGBExpansion(feed, refl, n_beams=n_beams, w_rel=w_rel,
                                      cone_margin=cone_margin, reg=reg,
                                      verbose=verbose, basis=ff_basis,
                                      edge_crescent=edge_crescent,
                                      crescent_order=crescent_order,
                                      goedge_takenaka=goedge_takenaka,
                                      goedge_crescent=goedge_crescent).build()
    else:
        exp = FeedGBExpansion(feed, refl, n_beams=n_beams, overlap=overlap,
                              theta_fill=theta_fill, reg=reg, verbose=verbose,
                              edge_crescent=edge_crescent,
                              crescent_order=crescent_order,
                              goedge_takenaka=goedge_takenaka,
                              goedge_crescent=goedge_crescent).build()

    S = directions(theta_scan_deg, phi_deg)
    if form == "literal":
        print("  注意: 使用字面式 (74) T*P_r + P_d; 在 S_a->0 的阴影边界处两项"
              "各自发散且数值上不能完全相消, 会给出 1e2 以上的假电平(仅供对照)")
    if form == "auto":
        print("  合成式: auto(远离阴影边界用字面式, |S_a|<0.7 切到稳定式)")
    if form == "crescent":
        print("  合成式: crescent(嫁接式, 仅对照): 在现有 fall 上减 S_ratio, 需加窗+上限")
    if form == "analytic":
        print("  合成式: analytic(纯解析): P_r*([1+erf(√c_eff·d_r)]/2 - S_ratio), "
              "无复根求解/无窗 (REPORT A.27)")
    if form == "goedge":
        print(f"  合成式: goedge(GO 反射 + PO 边缘): {'Takenaka erfc 分支式' if goedge_takenaka else '紧致 erf 式'}"
              f"{' + 月牙' if goedge_crescent else ''} (REPORT A.28)")
    if phi1se_mode != "newton":
        print(f"  边缘点 phi1se 解法: {phi1se_mode}"
              f"({'四次闭式, 无 (B13) 近似' if phi1se_mode in ('quartic_pure', 'closed') else phi1se_mode})")
    # ---- 合成：主曲线 + 可选的额外曲线，**一次遍历**同时累积（REPORT A.30）----
    extra_forms = tuple(str(f).lower() for f in (extra_forms or ()))
    extra_forms = tuple(f for f in extra_forms if f != str(form).lower())
    if extra_forms:
        outs, info = gbma_far_field_multi(exp, S, [form] + list(extra_forms),
                                          phi1se_mode=phi1se_mode)
        E = outs[form]
    else:
        outs = {}
        E, info = gbma_far_field(exp, S, form=form, phi1se_mode=phi1se_mode)
    P_total = compute_feed_total_power(feed)
    U = np.sum(np.abs(E) ** 2, axis=-1)
    dBi = 10.0 * np.log10(4.0 * np.pi * U / P_total + 1e-30)
    extra_dBi = {}
    for _f in extra_forms:
        _U = np.sum(np.abs(outs[_f]) ** 2, axis=-1)
        extra_dBi[_f] = 10.0 * np.log10(4.0 * np.pi * _U / P_total + 1e-30)
    if verbose and info.get("n_transition", 0) > 0:
        print(f"  提示: 有 {info['n_transition']} 个 (束,方向) 落在 |S_a|<1 的过渡区"
              f"(阴影边界); 这些点必须用稳定式, 字面式在此发散")
    if verbose and info.get("n_clip", 0) > 0:
        print(f"  提示: 有 {info['n_clip']} 个 (束,方向) 落在 |Im S_a| >~ 26.7 的远旁瓣区, "
              f"其复 erf 溢出已被渐近极限截断(该处 |P_r| 已下溢, 物理上就是 0)")
    info.update(dict(expansion=exp, reflector=refl,
                     max_incidence_deg=exp.max_incidence_deg,
                     theta_cover=np.degrees(exp.theta_cover),
                     dtheta=np.degrees(exp.dtheta), w0=exp.w0, b=exp.b,
                     overlap=overlap, theta_fill=theta_fill,
                     decomp=decomp, w_rel=w_rel,
                     edge_crescent=edge_crescent, crescent_order=crescent_order,
                     goedge_takenaka=goedge_takenaka,
                     goedge_crescent=goedge_crescent,
                     extra_dBi=extra_dBi,
                     n_inside=sum(1 for b in exp.beams if b.inside)))
    if return_expansion:
        return E, dBi, info
    return E, dBi, info

