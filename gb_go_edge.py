# -*- coding: utf-8 -*-
"""goedge —— GO 反射 + PO 边缘衍射（参照 Rieckmann et al. 2002 的 DGBA 结构）

与现有两条路径的关系
--------------------
    stable   : P_r(逐方向 SPP + 逐方向面法向/束幅) × ½[1+erf(S_a)]   —— 贵，且 S_a 要解 (B7)
    analytic : P_r(逐方向 SPP) × [½[1+erf(Z)] − S_ratio]             —— 无复根，但 P_r 仍逐方向算
    goedge   : P_GO(每束冻结, 逐方向只乘一个标量 exp) × [½[1+erf(Z)] − S_ratio]

核心差别只有一处：**反射本体不再逐方向求鞍点**。
取镜面方向 S_spec，把 f 的二次型 A=[[a0,cc],[cc,a1]] 与矢量幅度 F0=fbar(0,0,S_spec)
都**冻结在反射点**，于是对任意观察方向 ŝ（局部系分量），反射本体是闭式

    b   = (ŝ_x - c31, ŝ_y - c32)
    f_s = (a1 b_x² + a0 b_y² - 2 b_x b_y cc) / (4 (a0 a1 - cc²))
    P_GO(ŝ) = F0 · (-π/(w1 w2)) · exp(+jk f_s),   w1 w2 = jk sqrt(a0a1-cc²)

这正是"高斯束的解析远场"：二次型的全平面积分与鞍点求值**严格相同**（高斯积分精确），
差别只在矢量幅度取在反射点还是鞍点（跨足迹的幅度/极化变化，A.19/A.25.5 实测 ~1-2%）。

边缘衍射沿用 PO 半屏闭式（chou1997 (62)/(63) 的直边形式 + 曲边月牙项，REPORT A.27）：

    Z = sqrt(c_eff) d_r - jk b_eff/(2 sqrt(c_eff)),  S_ratio = e^{-Z²} λ M2 / sqrt(π)
    T = ½[1 + erf(Z)] - S_ratio,   P = P_GO · T

其中 c_eff = jk(a_ξξ - a_ξη²/a_ηη)、b_eff = b_ξ - a_ξη b_η/a_ηη 是旋到 (e_r,e_φ) 后的
Schur 补；d_r, κ 只用**每束一次**的实几何（原点 = 束的反射点）。

代价：没有复根求解、没有逐方向面法向、没有逐方向 det 比。
每 (束,方向) 只剩 ~30 次浮点 + 1 个复 exp + 1 个复 erf。
"""
import numpy as np
from scipy.special import erf, erfc

from gb_single import SQRTPI, spim, srep


class GoReflectionFast:
    """**纯 GO 反射的精简实现**（`form="go2"`）—— 按整个展开建一次，全向量化

    只用 GO 反射那一半（Rieckmann et al. 2002 的"几何波"）：

        P_GO(ŝ) = F0 · (jπ / (k √det A)) · exp(jk f_s(ŝ))

    其中 A 是 f 的二次型（**随方向**，见 A.28.2）：
        a0(ŝ) = A0 + G0·(S_z − c33),  a1(ŝ) = A1 + G1·(S_z − c33),  c = const
        f_s = (a1 b0² + a0 b1² − 2 b0 b1 c) / (4 det A),  b = ŝ − ŝ_spec
    F0 = 反射点处的矢量幅度（fbar 在原点、镜面方向求值，逐束一次）。

    与 `GoEdgeBeam.Pr_only` 的区别：
      * **不建边缘几何**（不算 d_r/κ/e_r/e_φ/月牙）；
      * 把 (束, 方向) 拉平成 (N, M) 大数组，numpy 调用次数从"每束 ~40 次"
        降到"总共 ~15 次" ⟹ 实测约 30 ms（逐束版约 190 ms）。
    """

    __slots__ = ("exp", "k", "N", "Rl", "O", "c31", "c32", "c33",
                 "A0", "A1", "cc", "G0", "G1", "F0")

    def __init__(self, expansion, k=None):
        import gb_single as gs
        self.exp = expansion
        self.k = float(expansion.k if k is None else k)
        k = self.k
        beams = expansion.beams
        self.N = len(beams)
        self.Rl = np.stack([np.stack([b.lm.xhat, b.lm.yhat, b.lm.zhat]) for b in beams])
        self.O = np.array([b.O for b in beams])
        Cs = np.array([np.asarray(b.solver.c[2], dtype=float) for b in beams])   # (N,3)
        self.c31, self.c32, self.c33 = Cs[:, 0], Cs[:, 1], Cs[:, 2]
        A0 = np.empty(self.N, dtype=complex)
        A1 = np.empty(self.N, dtype=complex)
        cc = np.empty(self.N, dtype=complex)
        for n, b in enumerate(beams):
            a0, a1, _, _, c = b.solver.coeffs(Cs[n])      # 镜面方向: b0 = b1 = 0
            A0[n], A1[n], cc[n] = a0, a1, c
        self.A0, self.A1, self.cc = A0, A1, cc
        self.G0 = np.array([0.5 / b.lm.R1 for b in beams])
        self.G1 = np.array([0.5 / b.lm.R2 for b in beams])
        # F0：原点处 z_i=0（det 比=1）、局部法向 nvl=(0,0,1)
        # ⚠ 镜面方向的 z 分量要镜像: (c31,c32,−c33)，见 A.28.2 —— 取错会让矢量幅度
        #   在 z 分量翻号（|F| 不变、方向变），91 束相干求和后峰值差 ~0.47 dB。
        H0 = np.array([np.asarray(b.beam.H0, dtype=complex) for b in beams])
        nvl = np.zeros((self.N, 3), dtype=complex)
        nvl[:, 2] = 1.0
        J = np.cross(nvl, H0)                                  # (17)
        Ssp = Cs * np.array([1.0, 1.0, -1.0])
        self.F0 = (1j * k * gs.Z0 / (2.0 * np.pi)) * np.cross(Ssp, np.cross(Ssp, J))

    def far_field(self, S, origin=None, _T=None):
        """返回 Omega (M,3)（未乘 _IMPEDANCE_SCALE）；_T 为可选的 (N,M) 截断因子"""
        S = np.atleast_2d(np.asarray(S, dtype=float))
        k = self.k
        if origin is None:
            origin = np.zeros(3)
        Sl = np.einsum("nkj,mj->nmk", self.Rl, S)              # 局部系分量 (N,M,3)
        b0 = Sl[..., 0] - self.c31[:, None]
        b1 = Sl[..., 1] - self.c32[:, None]
        dz = Sl[..., 2] - self.c33[:, None]
        a0 = self.A0[:, None] + self.G0[:, None] * dz
        a1 = self.A1[:, None] + self.G1[:, None] * dz
        cc = self.cc[:, None]
        with np.errstate(all="ignore"):
            den = a0 * a1 - cc * cc
            f_s = (a1 * b0 ** 2 + a0 * b1 ** 2 - 2.0 * b0 * b1 * cc) / (4.0 * den)
            # 与 ChouPathakSingle.Pr_only 逐位一致的分支写法：
            #   scalar = −π/(w1·w2),  w1=spim(−jk a1), w2=spim(−jk den/a1)
            # （不能写成 jπ/(k√den)：spim 对两个因子各自取 Im≥0，整体符号会翻）
            w1 = spim(-1j * k * a1)
            w2 = spim(-1j * k * (den / a1))
            scalar = (-np.pi / (w1 * w2)) * np.exp(1j * k * f_s)
            if _T is not None:
                scalar = scalar * _T
            P_loc = scalar[..., None] * self.F0[:, None, :]            # (N,M,3)
            P_glb = np.einsum("nmj,njk->nmk", P_loc, self.Rl)           # -> 全局系
            piston = np.exp(1j * k * (S @ (self.O - origin).T)).T       # (N,M)
            out = np.einsum("nmk,nm->mk", P_glb, piston)
        bad = ~np.isfinite(out).all(axis=-1)
        if np.any(bad):
            out = np.where(bad[..., None], 0.0, out)
        return out


class GoEdgeFast(GoReflectionFast):
    """**goedge 的扁平化加速版**（`form="goedge2"`）：GO 反射 + Takenaka 边缘项

    与 `GoReflectionFast`（纯 GO）唯一差别：在扁平化的 (N, M) 数组上再乘一个
    截断因子 T（Takenaka (25)+(26) 的"几何波 + 边界绕射波"，可选月牙）：

        Z = √c_eff·d_r − jk b_eff/(2√c_eff)
        T_st = E(Z) − (sgn/2)·erfc(sgn·Z)
        T = T_st − S_ratio            （crescent=True 时）
        P = P_GO(ŝ) · T(ŝ)

    其中 c_eff, b_eff 是二次型/线性项旋到 (e_r, e_φ) 后的 Schur 补。
    与逐束版 `GoEdgeBeam.Ps` **数值等价**（只差求和顺序），但：
      * 不逐方向调 `fbar`（沿用冻结幅度 F0）；
      * 每方向只做一次 (N,M) 的向量运算；
      * `d_r/κ/e_r/e_φ` 每束一次（与逐束版相同）。
    """

    __slots__ = ("d_r", "kappa", "e_r", "e_p", "ok", "order", "crescent")

    def __init__(self, expansion, k=None, crescent=True, order=1):
        super().__init__(expansion, k=k)
        self.crescent = bool(crescent)
        self.order = int(order)
        N = self.N
        d_r = np.zeros(N)
        kappa = np.zeros(N)
        e_r = np.zeros((N, 2))
        e_p = np.zeros((N, 2))
        ok = np.zeros(N, dtype=bool)
        for n, b in enumerate(expansion.beams):
            lm = b.lm
            good = bool(np.isfinite(lm.d) and lm.d > 0.0
                        and not bool(getattr(lm, "d_imag", False)))
            ok[n] = good
            if not good:
                e_r[n] = (1.0, 0.0)
                e_p[n] = (0.0, 1.0)
                continue
            gm = b.solver.rim_nearest(0.0, 0.0)
            d_r[n] = float(np.real(gm["d_r"]))
            kappa[n] = float(np.real(gm["kappa"]))
            e_r[n] = np.asarray(gm["er"], dtype=float).reshape(-1)[:2]
            e_p[n] = np.asarray(gm["ep"], dtype=float).reshape(-1)[:2]
        self.d_r, self.kappa, self.e_r, self.e_p, self.ok = d_r, kappa, e_r, e_p, ok

    def T_factor(self, S):
        """返回截断因子 T (N, M) 与 Z (N, M)"""
        S = np.atleast_2d(np.asarray(S, dtype=float))
        k = self.k
        Sl = np.einsum("nkj,mj->nmk", self.Rl, S)
        b0 = Sl[..., 0] - self.c31[:, None]
        b1 = Sl[..., 1] - self.c32[:, None]
        dz = Sl[..., 2] - self.c33[:, None]
        a0 = self.A0[:, None] + self.G0[:, None] * dz
        a1 = self.A1[:, None] + self.G1[:, None] * dz
        cc = self.cc[:, None]
        ex, ey = self.e_r[:, 0:1], self.e_r[:, 1:2]
        px_, py_ = self.e_p[:, 0:1], self.e_p[:, 1:2]
        with np.errstate(all="ignore"):
            a_rr = a0 * ex * ex + 2.0 * cc * ex * ey + a1 * ey * ey
            a_rp = a0 * ex * px_ + cc * (ex * py_ + ey * px_) + a1 * ey * py_
            a_pp = a0 * px_ * px_ + 2.0 * cc * px_ * py_ + a1 * py_ * py_
            c_eff = 1j * k * (a_rr - a_rp ** 2 / a_pp)
            sq = srep(c_eff)
            b_r = b0 * ex + b1 * ey
            b_p = b0 * px_ + b1 * py_
            b_eff = b_r - a_rp * b_p / a_pp
            d_r = np.where(self.ok, self.d_r, 0.0)[:, None]
            Z = sq * d_r - 1j * k * b_eff / (2.0 * sq)
            # Takenaka (25)+(26): 几何波 E(Z) + 边界绕射波 D_u
            lit = np.real(Z) >= 0.0
            sgn = np.where(lit, 1.0, -1.0)
            T = np.where(lit, 1.0, 0.0) - 0.5 * sgn * erfc(sgn * Z)
            if self.crescent:
                kappa = np.where(self.ok, self.kappa, 0.0)[:, None]
                sq_ = np.where(self.ok[:, None], sq, 1.0 + 0.0j)
                lam = sq_ * kappa / 2.0
                s2 = 1.0 / (2.0 * (1j * k * a_pp))
                acc = lam * s2
                if self.order >= 2:
                    acc = acc + lam ** 2 * (2.0 * Z) * (3.0 * s2 ** 2) / 2.0
                S_ratio = np.exp(-Z * Z) * acc / SQRTPI
                S_ratio = np.where(np.isfinite(S_ratio), S_ratio, 0.0)
                T = T - S_ratio
            T = np.where(np.isfinite(T), T, 1.0)
        return T, Z

    def far_field(self, S, origin=None):
        T, _ = self.T_factor(S)
        P = super().far_field(S, origin, _T=T)
        return P


class GoEdgeBeam:
    """一束的 GO 反射 + PO 边缘衍射（每束一次 setup，之后逐方向闭式）"""

    __slots__ = ("lm", "beam", "k", "spp", "ok", "Sspec", "c31", "c32", "c33",
                 "a0", "a1", "cc", "den", "scalar0", "F0", "d_r", "kappa",
                 "e_r", "e_p", "a_rp", "a_pp", "c_eff", "sq", "order",
                 "takenaka", "conjugate", "crescent")

    def __init__(self, lm, beam, k=None, order=1, takenaka=True, conjugate=False,
                 crescent=True):
        import gb_single as gs
        self.lm, self.beam = lm, beam
        self.spp = gs.ChouPathakSingle(lm, beam, k=k)
        self.k = self.spp.k
        self.order = int(order)
        # takenaka=True (**默认**): 边缘项用 Takenaka&Fukumitsu(1982) (25)+(26) 的分支式
        #   T_st = E(Z) − (sgn/2)erfc(sgn Z)  —— 几何波 + 边界绕射波
        # False: 紧致式 T_st = ½[1+erf(Z)]。两者代数恒等(实测相对差 4e-15)。
        self.takenaka = bool(takenaka)
        self.conjugate = bool(conjugate)
        # crescent: 是否再叠加"曲边/月牙"项 S_ratio (REPORT A.24/A.25)。
        #   Takenaka 的典范解本身就是**直线半屏**, 曲率是它之外的东西; 若实测不加更好,
        #   就说明 goedge 不需要它 (本路径只保留 Takenaka 的纯解析形式)。
        self.crescent = bool(crescent)
        self._setup()

    # ---- 每束一次 ----
    def _setup(self):
        spp, k, lm = self.spp, self.k, self.lm
        C = self.spp.c
        self.c31, self.c32, self.c33 = float(C[2][0]), float(C[2][1]), float(C[2][2])
        Ssp = np.asarray(C[2], dtype=float)          # 入射束轴(局部系)
        # ⚠ 反射方向要把入射轴关于切平面镜像: (c31,c32,-c33)。
        # 横向分量与入射轴相同(b0=b1=0, 故鞍点/Z 不受影响), 但 S_z 的符号会改变
        # Ŝ×(Ŝ×J) —— 取错会让矢量幅度在 z 分量翻号(实测最差束差 30%)。
        self.Sspec = np.array([Ssp[0], Ssp[1], -Ssp[2]], dtype=float)
        self.ok = bool(np.isfinite(lm.d) and lm.d > 0.0
                       and not bool(getattr(lm, "d_imag", False)))
        # ⚠ 二次型**不能**冻结: a0,a1 里的 (Sz-c33)/R 项随方向变化 30~90%,
        # 冻结会把反射束方向图变成"不聚焦"的宽束(实测 20° 处差 4 个量级).
        # 能冻结的只有**矢量幅度** F0(反射点处), 这是 GO 射线管近似的合理部分。
        a0, a1, b0, b1, cc = spp.coeffs(Ssp)         # 仅供边缘几何用(Ssp 处 b=0)
        self.a0, self.a1, self.cc = complex(a0), complex(a1), complex(cc)
        self.den = self.a0 * self.a1 - self.cc ** 2
        F0 = spp.fbar(np.zeros(1), np.zeros(1), self.Sspec[None, :])
        self.F0 = np.asarray(F0).reshape(-1)[:3]
        if not self.ok:
            self.d_r, self.kappa = 0.0, 0.0
            self.e_r = np.array([1.0, 0.0])
            self.e_p = np.array([0.0, 1.0])
            self.a_rp = self.a_pp = 0j
            self.c_eff = self.sq = 1.0 + 0.0j
            return
        gm = spp.rim_nearest(0.0, 0.0)               # 原点 = 束的反射点
        self.d_r = float(np.real(gm["d_r"]))
        self.kappa = float(np.real(gm["kappa"]))
        self.e_r = np.asarray(gm["er"], dtype=float).reshape(-1)[:2]
        self.e_p = np.asarray(gm["ep"], dtype=float).reshape(-1)[:2]

    # ---- 只算 GO 反射本体（Rieckmann 的"几何波"，无边缘项）----
    def Pr_only(self, S):
        """返回 GO 反射束的远场（冻结矢量幅度 F0 × 标量因子），**不算**边缘项

        这就是 Rieckmann et al. (2002) 里"反射按 GO 处理"的那一半，
        也是本模块 form="go2" 对应的量。每 (束,方向) 只有几次乘加 + 1 个复 exp。
        """
        S = np.atleast_2d(np.asarray(S, dtype=float))
        a0, a1, b0, b1, cc, xs, ys, fs, den = self.spp._cache(S)
        with np.errstate(all="ignore"):
            w1 = spim(-1j * self.k * a1)
            w2 = spim(-1j * self.k * (den / a1))
            scalar = (-np.pi / (w1 * w2)) * np.exp(1j * self.k * fs)
            P = scalar[..., None] * self.F0
        bad = ~np.isfinite(P).all(axis=-1)
        if np.any(bad):
            P = np.where(bad[..., None], 0.0, P)
        return P

    # ---- 逐方向：反射本体走 SPP(二次型随方向), 矢量幅度冻结; 边缘用闭式 ----
    def Ps(self, S):
        S = np.atleast_2d(np.asarray(S, dtype=float))
        k = self.k
        spp = self.spp
        a0, a1, b0, b1, cc, xs, ys, fs, den = spp._cache(S)
        with np.errstate(all="ignore"):
            # (1) 反射本体: 与精确 SPP 同一个标量因子, 只把矢量幅度换成 F0
            w1 = spim(-1j * k * a1)
            w2 = spim(-1j * k * (den / a1))
            scalar = (-np.pi / (w1 * w2)) * np.exp(1j * k * fs)
            P_go = scalar[..., None] * self.F0
            # (2) PO 边缘衍射: Takenaka & Fukumitsu (1982) 的"几何波 + 边界绕射波"结构
            #     U^K = U_G + U_D,  U_G = E(x-x_s)U_i,  U_D = D_u[..]e^{jk s(Q_d)}U_i(Q_d)
            #     关键差别: 绕射波从**边缘上的衍射点 Q_d** 辐射, 不是从束心.
            #     因此相对束心要乘 e^{+jk ŝ·r_e}（r_e = 束心->最近边缘点）。
            #     用 ½[1+erf(Z)] = E(Z) - (sgn/2)erfc(sgn·Z) 把它拆开, 再加相位。
            if self.ok:
                ex, ey = self.e_r
                px_, py_ = self.e_p
                a_rr = a0 * ex * ex + 2 * cc * ex * ey + a1 * ey * ey
                a_rp = a0 * ex * px_ + cc * (ex * py_ + ey * px_) + a1 * ey * py_
                a_pp = a0 * px_ * px_ + 2 * cc * px_ * py_ + a1 * py_ * py_
                c_eff = 1j * k * (a_rr - a_rp ** 2 / a_pp)
                sq = srep(c_eff)
                b_r = b0 * ex + b1 * ey
                b_p = b0 * px_ + b1 * py_
                b_eff = b_r - a_rp * b_p / a_pp
                Z = sq * self.d_r - 1j * k * b_eff / (2.0 * sq)
                if self.conjugate:
                    Z = np.conj(Z)
                # ---- 直边截断因子 T_st: 两种等价写法 (REPORT A.28.6) ----
                #   takenaka=True : Takenaka & Fukumitsu (1982) (25)+(26) 的分支式
                #       U^K = U_G + U_D,  U_G = E(x-x_s)·U_i,
                #       U_D = D_u·[q(z0)/q(z)]^{1/2}·e^{jk s(Q_d)}·U_i(Q_d)
                #       ⟹ T_st = E(Z) − (sgn/2)·erfc(sgn·Z),  sgn = sign(Re Z)
                #   takenaka=False: 紧致式 T_st = ½[1+erf(Z)]
                # 两者代数恒等（照亮 1−½erfc = ½[1+erf]；阴影 ½erfc(−Z) = ½[1+erf]），
                # 实测逐元素相对差 1.6e-15 ⟹ **结果与开关无关**，只是写法不同。
                # ⚠ 不要往这里加"束场在边缘点之比"或"边缘→远场相位": Takenaka 的
                #   U_i(Q_d) 与 U_i(P) 相对同一观察点归一，(26) 已含边缘→观察点的传播；
                #   额外相位等于重复计入，会破坏过渡区抵消（2–5° 由 0.44 恶化到 5.09 dB）。
                if self.takenaka:
                    lit = np.real(Z) >= 0.0                          # E(x − x_s)
                    sgn = np.where(lit, 1.0, -1.0)
                    T_st = np.where(lit, 1.0, 0.0) - 0.5 * sgn * erfc(sgn * Z)
                else:
                    T_st = 0.5 * (1.0 + erf(Z))
                # 月牙(曲边)项：可选。Takenaka 只处理直线边缘，曲率由此项补齐；
                #   goedge 若实测不需要，就保持 crescent=False（纯 Takenaka）。
                if self.crescent:
                    c_p = 1j * k * a_pp
                    lam = sq * self.kappa / 2.0
                    s2 = 1.0 / (2.0 * c_p)
                    acc = lam * s2
                    if self.order >= 2:
                        acc = acc + lam ** 2 * (2.0 * Z) * (3.0 * s2 ** 2) / 2.0
                    S_ratio = np.exp(-Z * Z) * acc / SQRTPI
                    S_ratio = np.where(np.isfinite(S_ratio), S_ratio, 0.0)
                    T = T_st - S_ratio
                else:
                    S_ratio = np.zeros(np.shape(Z), dtype=complex)
                    T = T_st
                T = np.where(np.isfinite(T), T, 1.0)
            else:
                Z = np.zeros(np.shape(b0), dtype=complex)
                S_ratio = np.zeros(np.shape(b0), dtype=complex)
                T = np.ones(np.shape(b0))
        P = P_go * T[..., None]
        bad = ~np.isfinite(P).all(axis=-1)
        if np.any(bad):
            P = np.where(bad[..., None], 0.0, P)
        return {"Ps": P, "Ps_stable": P, "Ps_auto": P, "Ps_crescent": P,
                "Ps_analytic": P, "Pr": P_go, "T": T, "T_analytic": T,
                "T_crescent": T, "S_ratio": S_ratio, "Sa": Z, "Z": Z,
                "d_r": np.full(np.shape(b0), self.d_r),
                "kappa_rim": np.full(np.shape(b0), self.kappa),
                "n_clip": 0, "n_near": 0, "n_transition": 0, "phi1se": None}
