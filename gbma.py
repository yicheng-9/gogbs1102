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

    __slots__ = ("O", "L", "dir", "theta", "lm", "beam", "solver", "coef", "inside")

    def __init__(self, O, L, dir, theta, lm, beam, inside):
        self.O, self.L, self.dir, self.theta = O, L, dir, theta
        self.lm, self.beam, self.inside = lm, beam, inside
        self.solver = gs.ChouPathakSingle(lm, beam)
        self.coef = 1.0 + 0.0j


class FeedGBExpansion:
    """把馈源的照射场展开成一族高斯波束(见模块 docstring)"""

    def __init__(self, feed, reflector, n_beams=91, overlap=1.2, theta_fill=1.0,
                 reg=1e-8, verbose=True):
        self.feed = feed
        self.k = float(feed.k)
        self.refl = reflector
        self.overlap = float(overlap)
        self.theta_fill = float(theta_fill)
        self.reg = float(reg)
        self.verbose = verbose

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

        self.rim = reflector.rim(1440)
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
            lm = gs.LocalModelFit(self.refl.parab, O, self.rim, verbose=False)
            R = float(np.linalg.norm(O - self.W))
            to_l = np.stack([lm.xhat, lm.yhat, lm.zhat])       # 行 = 局部基(全局分量)
            zi_l = to_l @ _normalize(d)
            pol_l = to_l @ self._h_hat(d)
            beam = gs.GaussianBeam(self.b, R, zi_l, pol_ref=pol_l, k=self.k)
            beams.append(GBBeam(O=O, L=R, dir=_normalize(d), theta=float(th), lm=lm,
                                beam=beam, inside=bool(self.refl.inside(O))))
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
        """点匹配: Σ_n c_n Ψ_n(O_m) = H_feed(O_m)  (Ψ 为单位轴上幅度的 GB)"""
        N = len(self.beams)
        Rls = [np.stack([b.lm.xhat, b.lm.yhat, b.lm.zhat]) for b in self.beams]
        A = np.zeros((3 * N, N), dtype=complex)
        for m, bm in enumerate(self.beams):
            for n, bn in enumerate(self.beams):
                psi = bn.beam.H(bn.lm.to_local(bm.O))     # 局部基
                A[3 * m:3 * m + 3, n] = psi @ Rls[n]      # -> 全局基
        rhs = np.zeros(3 * N, dtype=complex)
        for m, bm in enumerate(self.beams):
            rhs[3 * m:3 * m + 3] = self.feed_H(bm.O)
        A_unit, rhs_unit = A.copy(), rhs.copy()
        # 轻微 Tikhonov 正则, 防止格点过密时系数爆炸
        if self.reg > 0:
            scale = np.linalg.norm(A) / np.sqrt(max(A.size, 1))
            A = np.vstack([A, np.sqrt(self.reg) * scale * np.eye(N, dtype=complex)])
            rhs = np.concatenate([rhs, np.zeros(N, dtype=complex)])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        self.sol = sol
        self.sing_vals = np.linalg.svd(A_unit, compute_uv=False)
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
# 远场合成
# --------------------------------------------------------------------------- #
def directions(theta_deg, phi_deg):
    """观察方向(全局基), theta 从 +z 轴起算"""
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    ph = np.deg2rad(np.asarray(phi_deg, dtype=float))
    return np.stack([np.sin(th) * np.cos(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(th)], axis=-1)


def gbma_far_field(expansion, S, origin=None, stable=True, per_beam=False, k=None,
                   form=None):
    """合成远场 Ω(S) = Σ_n Rot·P̄^s_n(Rot^T S)·exp(+jk S·(O_n-O_0))

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
    _KEY = {"stable": "Ps_stable", "literal": "Ps", "auto": "Ps_auto"}
    if form not in _KEY:
        raise ValueError(f"form 只能是 {list(_KEY)}, 收到 {form!r}")
    out = np.zeros((nS, 3), dtype=complex)
    n_clip = 0
    n_tr = 0
    contrib = np.zeros((nS, len(expansion.beams)), dtype=complex) if per_beam else None
    for j, b in enumerate(expansion.beams):
        Rl = np.stack([b.lm.xhat, b.lm.yhat, b.lm.zhat])       # 行 = 局部基
        Sl = S @ Rl.T                                          # 全局 -> 局部 分量
        d = b.solver.Ps(Sl)
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
             "auto": "auto: |Sa|>1.5 用字面式, <0.7 用稳定式(平滑过渡)"}
    info = {"n_beams": len(expansion.beams), "n_clip": n_clip, "n_transition": n_tr,
            "form": _NAME[form], "form_key": form}
    if per_beam:
        info["per_beam"] = contrib
    return out * _IMPEDANCE_SCALE, info


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
                           phi_deg=0.0, return_expansion=False, form="auto"):
    """馈源分解 → 逐束反射 → 合成的总入口。

    参数
    ----
    feed   : feed_antenna.FeedAntenna
    cfg    : c.py 的 AntennaConfig (用到 freq/F/D/offset_x/offset_y)
    theta_scan_deg : 观察角数组(度), 从反射面 +z 轴起算
    n_beams        : 束数(吸附到最近六边形格点数), None -> cfg.gbma_n_beams
    overlap        : 格点角距 / GB 角半径, None -> cfg.gbma_overlap
    theta_fill     : 覆盖角 / 口径张角, None -> cfg.gbma_theta_fill
    reg            : 点匹配正则, None -> cfg.gbma_reg

    返回 (E (N,3) 复数, dBi (N,), info dict)
    """
    n_beams = cfg.gbma_n_beams if n_beams is None else n_beams
    overlap = getattr(cfg, "gbma_overlap", 1.2) if overlap is None else overlap
    theta_fill = getattr(cfg, "gbma_theta_fill", 1.02) if theta_fill is None else theta_fill
    reg = getattr(cfg, "gbma_reg", 1e-8) if reg is None else reg

    refl = ParabolicReflector(cfg.F, cfg.D, cfg.offset_x, cfg.offset_y)
    exp = FeedGBExpansion(feed, refl, n_beams=n_beams, overlap=overlap,
                          theta_fill=theta_fill, reg=reg, verbose=verbose).build()

    S = directions(theta_scan_deg, phi_deg)
    if form == "literal":
        print("  注意: 使用字面式 (74) T*P_r + P_d; 在 S_a->0 的阴影边界处两项"
              "各自发散且数值上不能完全相消, 会给出 1e2 以上的假电平(仅供对照)")
    if form == "auto":
        print("  合成式: auto(远离阴影边界用字面式, |S_a|<0.7 切到稳定式)")
    E, info = gbma_far_field(exp, S, form=form)
    P_total = compute_feed_total_power(feed)
    U = np.sum(np.abs(E) ** 2, axis=-1)
    dBi = 10.0 * np.log10(4.0 * np.pi * U / P_total + 1e-30)
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
                     n_inside=sum(1 for b in exp.beams if b.inside)))
    if return_expansion:
        return E, dBi, info
    return E, dBi, info

