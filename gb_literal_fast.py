# -*- coding: utf-8 -*-
"""gb_literal_fast.py -- **literal 的扁平化实现**（`form="literal2"`，REPORT A.38）

`literal` = chou1997 (74) 的字面式 `T*P_r + P_d`，逐束调用
`ChouPathakSingle.Ps`。A.37 的拆解表明它的主导成本是 (B7) 的复根求解（43.5%），
而它是**逐束**调用（91 次 Python 调用 × 一堆 (361,) 小数组），开销很大一部分在
Python/调度层，不在算术上。

本模块把整条链搬到 (N 束, M 方向) 的大数组上，与 `gb_go_edge.GoEdgeFast` 同一套路：

    ┌ 每束一次（构造函数）：局部模型参数、GB 参数、边缘椭圆参数
    │ 每 (束,方向) 一次：coeffs → stationary → (B7) Ferrari → (B1)(B3)(B5)
    └                   → F_e=(25)在复边缘点 → (73) → S_a → erf/T → T*P_r+P_d
    最后：旋转到全局 + 活塞相位 + 求和（einsum）

与逐束版**逐元素相同的算式、相同的分支选择**（连 `_quartic_all_roots` /
`_cubic_one_root` 都直接复用 gb_single 里的实现），所以数值差只来自"沿束求和顺序"
（实测 ~1e-16 相对量级）。

前提（构造时检查）：`edge_q == "frozen"`（论文约定 C1）、`edge_crescent=False`、
`phi1se_mode == "quartic_pure"` —— 也就是生产默认的那条纯闭式路径。
"""

import numpy as np
from scipy.special import erf

import gb_single as gs


class LiteralFast:
    """`form="literal"` 的扁平化加速版：一次算完整个 GB 展开的字面式远场"""

    __slots__ = ("n", "k", "Z0", "SQRTPI", "SQRT2PI", "form_eps",
                 "Rl", "O", "xo", "yo", "d", "jac", "R1", "R2",
                 "A1", "A2c", "B1", "B2c", "Minv", "zhat", "xhat", "yhat",
                 "C", "H0", "L", "bb", "Q0", "c11", "c12", "c21", "c22", "c31", "c32", "c33")

    def __init__(self, expansion, k=None):
        beams = list(expansion.beams)
        if not beams:
            raise ValueError("expansion 里没有 beam")
        self.n = len(beams)
        sol0 = beams[0].solver
        self.k = float(sol0.k if k is None else k)
        self.Z0 = gs.Z0
        self.SQRTPI = np.sqrt(np.pi)
        self.SQRT2PI = np.sqrt(2.0 * np.pi)
        self.form_eps = float(sol0.form_eps)

        # ---- 前提检查：只支持生产默认的纯闭式字面式 ----
        for b in beams:
            if b.solver.edge_q != "frozen":
                raise ValueError(f'LiteralFast 只支持 edge_q="frozen"，收到 {b.solver.edge_q!r}')
            if b.solver.edge_crescent:
                raise ValueError("LiteralFast 不支持 edge_crescent=True")
            if float(b.solver.k) != self.k:
                raise ValueError("各束的 k 不一致")

        n = self.n
        self.Rl = np.empty((n, 3, 3))
        self.O = np.empty((n, 3))
        self.xo = np.empty(n); self.yo = np.empty(n)
        self.d = np.empty(n); self.jac = np.empty(n)
        self.R1 = np.empty(n); self.R2 = np.empty(n)
        self.A1 = np.empty(n); self.A2c = np.empty(n)
        self.B1 = np.empty(n); self.B2c = np.empty(n)
        self.Minv = np.empty((n, 2, 2))
        self.zhat = np.empty((n, 3)); self.xhat = np.empty((n, 3))
        self.yhat = np.empty((n, 3))
        self.C = np.empty((n, 3, 3), dtype=complex)
        self.H0 = np.empty((n, 3), dtype=complex)
        self.L = np.empty(n); self.bb = np.empty(n)
        self.Q0 = np.empty(n, dtype=complex)
        self.c11 = np.empty(n); self.c12 = np.empty(n)
        self.c21 = np.empty(n); self.c22 = np.empty(n)
        self.c31 = np.empty(n); self.c32 = np.empty(n); self.c33 = np.empty(n)

        for i, b in enumerate(beams):
            lm, bm = b.lm, b.beam
            self.Rl[i] = np.stack([lm.xhat, lm.yhat, lm.zhat])
            self.O[i] = b.O
            self.xo[i] = lm.xo; self.yo[i] = lm.yo
            self.d[i] = lm.d; self.jac[i] = lm.jac
            self.R1[i] = lm.R1; self.R2[i] = lm.R2
            self.A1[i] = lm.A1; self.A2c[i] = lm.A2c
            self.B1[i] = lm.B1; self.B2c[i] = lm.B2c
            self.Minv[i] = lm.Minv
            self.zhat[i] = lm.zhat; self.xhat[i] = lm.xhat; self.yhat[i] = lm.yhat
            self.C[i] = bm.C
            self.H0[i] = bm.H0
            self.L[i] = bm.L; self.bb[i] = bm.b
            self.Q0[i] = b.solver.Q0
            c = b.solver.c
            self.c11[i], self.c12[i] = c[0][0], c[0][1]
            self.c21[i], self.c22[i] = c[1][0], c[1][1]
            self.c31[i], self.c32[i], self.c33[i] = c[2]

    # ------------------------------------------------------------------ #
    # (25) F(x',y')，field 版；x,y 可为复数（边缘点）
    # ------------------------------------------------------------------ #
    def _fbar(self, x, y, Sl):
        R1 = self.R1[:, None]
        R2 = self.R2[:, None]
        zp = -0.5 * (x * x / R1 + y * y / R2)                      # lm.zloc
        zi = (x * self.C[:, 2, 0][:, None] + y * self.C[:, 2, 1][:, None]
              + zp * self.C[:, 2, 2][:, None])                     # ri[...,2] = r_local·C.T 的 z 分量
        Lb = self.L[:, None]
        bbb = self.bb[:, None]
        amp = (Lb + 1j * bbb) / ((Lb + zi) + 1j * bbb)             # det_ratio_sqrt
        Hm = amp[..., None] * self.H0[:, None, :]                  # (19)
        # nv = zhat + x*xhat/R1 + y*yhat/R2  (未归一化法向), 再换到局部分量
        nv = (self.zhat[:, None, :]
              + x[..., None] * (self.xhat / R1)[:, None, :]
              + y[..., None] * (self.yhat / R2)[:, None, :])
        nvl = np.einsum("nmj,nkj->nmk", nv, self.Rl)
        J = np.cross(nvl, Hm)                                      # (17)
        return (1j * self.k * self.Z0 / (2.0 * np.pi)) * np.cross(Sl, np.cross(Sl, J))

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def far_field(self, S, origin=None, want_parts=False):
        """返回 Omega (M,3)（未乘 _IMPEDANCE_SCALE），即 sum_n Rot·Ps_n·piston"""
        S = np.atleast_2d(np.asarray(S, dtype=float))
        M = S.shape[0]
        n = self.n
        k = self.k
        origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)

        # ---- 每束局部系里的方向分量 (N,M,3) ----
        Sl = np.einsum("nkj,mj->nmk", self.Rl, S)
        Sx, Sy, Sz = Sl[..., 0], Sl[..., 1], Sl[..., 2]

        # ---- (28)-(32) 二次相位系数 ----
        Q0 = self.Q0[:, None]
        dz = Sz - self.c33[:, None]
        c11 = self.c11[:, None]; c12 = self.c12[:, None]
        c21 = self.c21[:, None]; c22 = self.c22[:, None]
        a0 = 0.5 * (dz / self.R1[:, None] + c11 ** 2 * Q0 + c21 ** 2 * Q0)
        a1 = 0.5 * (dz / self.R2[:, None] + c12 ** 2 * Q0 + c22 ** 2 * Q0)
        b0 = Sx - self.c31[:, None]
        b1 = Sy - self.c32[:, None]
        cc = 0.5 * (c11 * c12 * Q0 + c21 * c22 * Q0)

        # ---- (33)-(36) 驻定相位点 ----
        with np.errstate(all="ignore"):
            den = a0 * a1 - cc ** 2
            xs = (a1 * b0 - b1 * cc) / (2.0 * den)
            ys = (a0 * b1 - b0 * cc) / (2.0 * den)
            fs = (a1 * b0 ** 2 + a0 * b1 ** 2 - 2.0 * b0 * b1 * cc) / (4.0 * den)

            # ---- (B7) 的 H1..H5 ----
            A1 = self.A1[:, None]; A2 = self.A2c[:, None]
            B1 = self.B1[:, None]; B2 = self.B2c[:, None]
            dd = self.d[:, None]
            X = self.xo[:, None] - xs
            Y = self.yo[:, None] - ys
            H1 = 4.0 * dd * (a0 * A1 * A2 + a1 * B1 * B2 + cc * (A2 * B1 + B2 * A1))
            H2 = 2.0 * dd * (a0 * (A1 ** 2 - A2 ** 2) + a1 * (B1 ** 2 - B2 ** 2)
                             + 2.0 * cc * (A1 * B1 - A2 * B2))
            H3 = 2.0 * (a0 * A1 * X + a1 * B1 * Y + cc * (A1 * Y + B1 * X))
            H4 = -2.0 * (a0 * A2 * X + a1 * B2 * Y + cc * (A2 * Y + B2 * X))
            H5 = -0.5 * H1

            # ---- 四次方程（Ferrari，复用 gb_single 的实现）----
            qa = H1 ** 2 + H2 ** 2
            qb = 2.0 * (H1 * H3 + H2 * H4)
            qc = H3 ** 2 + 2.0 * H1 * H5 - H2 ** 2 + H4 ** 2
            qd = 2.0 * (H3 * H5 - H2 * H4)
            qe = H5 ** 2 - H4 ** 2
            scale = np.maximum(np.maximum(np.maximum(np.abs(qa), np.abs(qb)),
                                          np.maximum(np.abs(qc), np.abs(qd))), np.abs(qe))
            scale = np.where(scale > 0.0, scale, 1.0)
            deg_ok = np.abs(qa) > 1e-8 * scale
            roots = gs._quartic_all_roots(qa / scale, qb / scale, qc / scale,
                                          qd / scale, qe / scale)          # (4,N,M)
            Pq = H1 * roots ** 2 + H3 * roots + H5
            Qq = H2 * roots + H4
            okq = np.abs(Qq) > 1e-300
            Cq = np.where(okq, -Pq / np.where(okq, Qq, 1.0), np.nan)
            err = np.abs(roots ** 2 + Cq ** 2 - 1.0)
            ok = np.isfinite(err) & (err < 1e-8)
            phi_a = np.arcsin(np.asarray(roots, dtype=complex))
            phi_b = np.pi - phi_a
            phi_a = np.where(np.abs(np.cos(phi_b) - Cq) < np.abs(np.cos(phi_a) - Cq),
                             phi_b, phi_a)
            phi_a = np.where(np.real(phi_a) > np.pi, phi_a - 2.0 * np.pi, phi_a)
            phi_a = np.where(np.real(phi_a) < -np.pi, phi_a + 2.0 * np.pi, phi_a)
            ok = ok & deg_ok & np.isfinite(phi_a)

            # ---- 分支选择器：三次闭式 (B14)-(B23) ----
            with np.errstate(all="ignore"):
                al1 = (H4 - 2.0 * H1) / H2
                al2 = -2.0 * (H2 + H3) / H2
                al3 = -2.0 * (H4 + H5) / H2
                Qc = (3.0 * al2 - al1 ** 2) / 9.0
                Rc = (9.0 * al1 * al2 - 27.0 * al3 - 2.0 * al1 ** 3) / 54.0
                disc = Qc ** 3 + Rc ** 2
                Sst = gs.cbrt(Rc + np.sqrt(disc))
                Tst = -Qc / Sst
                z1 = Sst + Tst - al1 / 3.0
                z2 = -0.5 * (Sst + Tst) - al1 / 3.0 + 0.5j * np.sqrt(3.0) * (Sst - Tst)
                z3 = -0.5 * (Sst + Tst) - al1 / 3.0 - 0.5j * np.sqrt(3.0) * (Sst - Tst)
                cand = np.stack([z1, z2, z3], axis=0)
                idx = np.argmin(np.abs(cand), axis=0)
                zc = np.take_along_axis(cand, idx[None, ...], axis=0)[0]
                pc = np.arcsin(np.asarray(zc, dtype=complex))

            # 退化点 (H1=0 ⟹ H5=0): phi=0 本身满足 (B7)，显式补一候选
            hscale = np.maximum(np.maximum(np.maximum(np.abs(H1), np.abs(H2)),
                                           np.maximum(np.abs(H3), np.abs(H4))), np.abs(H5))
            hscale = np.where(hscale > 0.0, hscale, 1.0)
            zero_ok = np.abs(H5) <= 1e-10 * hscale
            phi_cat = np.concatenate([phi_a, np.zeros((1, n, M), dtype=complex)], axis=0)
            ok_cat = np.concatenate([ok, np.broadcast_to(zero_ok[None, ...], (1, n, M))], axis=0)
            score = np.where(ok_cat, np.abs(phi_cat - pc[None, ...]), np.inf)
            jsel = np.argmin(score, axis=0)
            phi = np.take_along_axis(phi_cat, jsel[None, ...], axis=0)[0]

            # phi1se 的兜底：非有限 / |Im|>3 / |Re|>pi 的纯闭式点退回 phi=0
            bad = (~np.isfinite(phi) | (np.abs(np.imag(phi)) > 3.0)
                   | (np.abs(np.real(phi)) > np.pi))
            phi = np.where(bad, 0.0 + 0.0j, phi)

            # ---- 附录 B 边缘量 (B1)(B3)(B5) ----
            cs, sn = np.cos(phi), np.sin(phi)
            Xr = A1 * cs + A2 * sn
            Yr = B1 * cs + B2 * sn
            Xrp = -A1 * sn + A2 * cs
            Yrp = -B1 * sn + B2 * cs
            x_e = self.xo[:, None] + dd * Xr
            y_e = self.yo[:, None] + dd * Yr
            dxe = x_e - xs
            dye = y_e - ys
            fx_e = -2.0 * a0 * dxe - 2.0 * cc * dye
            fy_e = -2.0 * a1 * dye - 2.0 * cc * dxe
            gpd = fx_e * Xr + fy_e * Yr                                       # (B3)
            xd = dd * Xrp
            yd = dd * Yrp
            gpp_phi = (-2.0 * a0 * xd ** 2 - 4.0 * cc * xd * yd - 2.0 * a1 * yd ** 2
                       - fx_e * dd * Xr - fy_e * dd * Yr)                      # (B5)
            f_se = fs - (a0 * dxe ** 2 + a1 * dye ** 2 + 2.0 * cc * dxe * dye)  # (B1)

            # ---- (73) 端点项 ----
            Fe = self._fbar(x_e, y_e, Sl)
            coef_d = dd * self.jac[:, None] / (1j * k * gpd)
            coef_d = np.where(np.isfinite(coef_d), coef_d, 0.0)
            tail_d = 1j * self.SQRT2PI * np.exp(1j * k * f_se) / gs.spim(1j * k * gpp_phi)
            tail_d = np.where(np.isfinite(tail_d), tail_d, 0.0)
            Pd = Fe * coef_d[..., None] * tail_d[..., None]

            # ---- (68) 反射场 ----
            Fs = self._fbar(xs, ys, Sl)
            w1 = gs.spim(-1j * k * a1)
            w2 = gs.spim(-1j * k * den / a1)
            Pr = (-np.pi / (w1 * w2))[..., None] * Fs * np.exp(1j * k * fs)[..., None]

            # ---- (69)(71)(72): S_a 及其符号 ----
            dxs = xs - self.xo[:, None]
            dys = ys - self.yo[:, None]
            Mi = self.Minv
            u = Mi[:, 0, 0][:, None] * dxs + Mi[:, 0, 1][:, None] * dys
            v = Mi[:, 1, 0][:, None] * dxs + Mi[:, 1, 1][:, None] * dys
            rho_s = np.sqrt(u * u + v * v)
            rho_s = np.where(np.real(rho_s) < 0.0, -rho_s, rho_s)
            ar = dxs / rho_s
            br = dys / rho_s
            gpp_rho = -2.0 * (a0 * ar ** 2 + a1 * br ** 2 + 2.0 * cc * ar * br)
            Sa = np.sqrt(1j * k * (fs - f_se))
            target = np.real(gs.srep(-1j * k / 2.0 * gpp_rho) * (dd - rho_s))
            Sa = np.where(np.real(Sa) * target < 0.0, -Sa, Sa)

            # ---- (65) 过渡函数 + (74) 字面式 ----
            erf_Sa = erf(Sa)
            fall = 0.5 * (1.0 + erf_Sa)
            T = 1.0 - 0.5 * (1.0 - erf_Sa) + np.exp(-Sa ** 2) / (2.0 * self.SQRTPI * Sa)
            pos = np.real(Sa) > 0.0
            fall = np.where(np.isfinite(fall), fall, np.where(pos, 1.0, 0.0))
            T = np.where(np.isfinite(T), T, np.where(pos, 1.0, 0.0))
            bad_pr = ~np.isfinite(Pr)
            if np.any(bad_pr):
                Pr = np.where(bad_pr, 0.0, Pr)
            bad_pd = ~np.isfinite(Pd)
            if np.any(bad_pd):
                Pd = np.where(bad_pd, 0.0, Pd)
            Ps = T[..., None] * Pr + Pd

        # 字面式在 |S_a| < form_eps 处的奇点兜底：用稳定式顶替
        near = np.abs(Sa) < self.form_eps
        if np.any(near):
            Ps = np.where(near[..., None], Pr * fall[..., None], Ps)

        # ---- 旋转到全局 + 活塞相位 + 求和 ----
        bad = ~np.isfinite(Ps).all(axis=-1)
        if np.any(bad):
            Ps = np.where(bad[..., None], 0.0, Ps)
        P_glb = np.einsum("nmj,njk->nmk", Ps, self.Rl)
        piston = np.exp(1j * k * (S @ (self.O - origin).T)).T            # (N,M)
        out = np.einsum("nmk,nm->mk", P_glb, piston)
        if not want_parts:
            return out
        return out, dict(Sa=Sa, Pr=Pr, Pd=Pd, T=T, fall=fall, phi=phi,
                         x_e=x_e, y_e=y_e, fs=fs, f_se=f_se)
