# -*- coding: utf-8 -*-
r"""GB 展开 —— **farfield 版**（在馈源远场方向图上点匹配，chou2001 §II）。

本文件是把用户提供的 `gb_expansion_farfield.py` 计划移植到**当前工作区**的自包含版本：
原文件 import 了 `gbma_common`（PI/normalize/orthobasis/cone_dir/hex_lattice）和
`gb_expansion_near`（GBBeam），这两个模块在本工作区不存在，因此这里：
  * 几何小工具直接从 `gbma` 取（`_normalize/_orthobasis/_cone_dir/GBBeam`）
  * `hex_lattice` 在本文件内实现（(10)(11) 的六边形交错格点）
⚠ 移植时保留了原文件那一步系数换算 `c = A_n·e^{-jkL}/L`，**它是错的**（少了 `jb`），
导致 A.17 里"远场匹配更差"的错误结论；A.18 已更正，本文件默认走正确换算。

与 `gbma.FeedGBExpansion`（near/surface 版）的区别只在"系数怎么定"和"w/b/Δθ 怎么选"：

| | near（gbma.FeedGBExpansion） | farfield（本文件） |
|---|---|---|
| 匹配点 | 反射面上的点 O_m | 馈源远区方向 Ω_q |
| 目标 | H_feed(O_m)（含 1/R、e^{-jkR}） | F(θ_q)（馈源远场标量方向图） |
| 格点 | 同心环，θ_cover/K | (10)(11) 六边形交错 |
| w | w0 = 2/(k θ_gb) 反推 | (19) 窗口内选，A_0 = πw² |
| b | b = k w0²/2 | (15) 小根，使 w(d_f)=w |
| Δθ | 由格点数摊分 | (13) Δθ = √(8ln4/(3kb)) |

**系数的换算（关键，见 REPORT.md 附录 A.18）**
每束的束腰都在馈源相位中心 W（`GaussianBeam(b, L_n, ...)`，L_n=|O_n-W|），
所有束共用同一个相位参考点，因此 chou2001 (2) 里的附加相位项在单格点情形下
恒等于零。把 (1) 严格取远区渐近极限（不丢 γ 的高阶项）得

    H_n(P) ≈ H0_n·(L_n + jb)·e^{jkL_n}·Ψ_n(γ)·e^{-jkR}/R,
    Ψ_n(γ) = exp[-(tanγ/θ_gb)²]/cosγ,   θ_gb = √(2/(kb)) = 2/(k w0)

故匹配方程组为（未知量直接取 H0_n，即反射点 O_n 处的入射场幅度）

    Σ_n H0_n·(L_n + jb)·e^{jkL_n}·Ψ_n(γ_qn) = F(θ_q)

注意 `(L_n + jb)` 不能简化成 `L_n`：L_n≈0.6~0.95 m、b≈0.15~0.65 m 时它带来
×1.02~×1.41 的幅度和 +9°~+45° 的相位（b=d_f 时正好 +45°），是本模块最容易写错的一步。
`basis="paraxial"` 可退回旁轴的 Ψ=exp[-(γ/θ_gb)²]（仅用于对比）。
"""
from __future__ import annotations

import numpy as np

import gb_single as gs
from gbma import (_normalize, _orthobasis, _cone_dir, GBBeam)

_PI = np.pi


# --------------------------------------------------------------------------- #
# chou2001 (13)-(19)
# --------------------------------------------------------------------------- #
def b_from_w(k, d_f, w):
    """(15) 的小根: b = q - sqrt(q^2 - d_f^2), q = k w^2/4（有理化避免相消）。

    w 取到 (16) 的边界 q=d_f 时 b=d_f（束腰在馈源、反射面恰好在一个瑞利距离上），
    此时 sqrt 的自变量理论上是 0，为避免浮点负数按 0 截断。
    """
    q = 0.25 * k * w * w
    disc = q ** 2 - d_f ** 2
    if disc < -1e-12 * d_f ** 2:               # (16)
        raise ValueError(f"chou2001 (16) 不满足: (k w^2/4)^2={q**2:.6g} < d_f^2={d_f**2:.6g};"
                         f" w={w:.6g} 太小(需 w >= 2 sqrt(d_f/k)={2*np.sqrt(d_f/k):.6g})")
    return d_f ** 2 / (q + np.sqrt(max(disc, 0.0)))


def delta_theta_from_b(k, b):
    """(13) Δθ ≈ sqrt(8 ln4 / (3 k b))"""
    return np.sqrt(8.0 * np.log(4.0) / (3.0 * k * b))


def w_window(k, d_f, D):
    """(19) 给出的 w 窗口 [w_lo, w_hi]: 4 d_f/k <= w^2 <= D^2/16。"""
    return float(np.sqrt(4.0 * d_f / k)), float(0.25 * D)


def check_13_19(k, d_f, D, w):
    """返回 (13)-(19) 的逐条诊断。"""
    w_lo, w_hi = w_window(k, d_f, D)
    q = 0.25 * k * w * w
    out = dict(w=w, w_lo=w_lo, w_hi=w_hi,
               eq16=bool(q ** 2 >= d_f ** 2 - 1e-12 * d_f ** 2),
               eq17=bool(np.pi * w * w >= 4.0 * np.pi * d_f / k),
               eq18=bool(np.pi * w * w <= np.pi * D * D / 16.0),
               eq19=bool(w_lo - 1e-12 <= w <= w_hi + 1e-12))
    if out['eq16']:
        b = b_from_w(k, d_f, w)
        out['b'] = b
        out['delta_theta'] = delta_theta_from_b(k, b)
        out['theta_gb'] = np.sqrt(2.0 / (k * b))
    return out


def hex_lattice(theta_cover, dtheta, dtheta_y=None):
    """chou2001 (10)(11) 的六边形交错格点（与用户 gbma_common.py 的实现逐行一致）。

    θy = q·Δθy，θx = l·Δθ，行距 Δθy = Δθ·sin60°；返回 [(theta, phi), ...]，
    theta 从极轴起算，phi 为绕轴的方位角。
    """
    if dtheta_y is None:
        dtheta_y = dtheta * np.sin(_PI / 3.0)
    qmax = int(np.floor(theta_cover / max(dtheta_y, 1e-12))) + 1
    pts = []
    for q in range(-qmax, qmax + 1):
        ty = q * dtheta_y
        if abs(ty) > theta_cover + 1e-12:
            continue
        tx_max = np.sqrt(max(theta_cover ** 2 - ty ** 2, 0.0))
        lmax = int(np.floor(tx_max / max(dtheta, 1e-12)))
        for l in range(-lmax, lmax + 1):
            tx = l * dtheta
            th = np.hypot(tx, ty)
            if th > theta_cover + 1e-12:
                continue
            ph = np.arctan2(ty, tx)
            pts.append((float(th), float(ph)))
    return pts


# --------------------------------------------------------------------------- #
# 远场版展开
# --------------------------------------------------------------------------- #
class GBExpansionFarField:
    """chou2001 §II 的远场 GB 展开。接口与 `gbma.FeedGBExpansion` 一致，
    可直接喂给 `gbma.gbma_far_field`。"""

    def __init__(self, feed, reflector, w=None, w_rel=0.0, theta_fill=1.0,
                 reg=1e-10, verbose=True, dtheta_scale=1.0, basis="exact"):
        self.feed = feed
        self.k = float(feed.k)
        self.refl = reflector
        self.reg = float(reg)
        self.verbose = verbose
        if basis not in ("exact", "paraxial"):
            raise ValueError(f"basis 只能是 'exact'/'paraxial'，收到 {basis!r}")
        self.basis = basis

        self.W = reflector.focus()
        self.ax = _normalize(reflector.center() - self.W)
        self.theta_rim = reflector.theta_extent(self.W, self.ax)
        self.theta_cover = float(theta_fill) * self.theta_rim

        # 参考极化（与 far_field_amplitude 的标量方向图配套）
        xg = np.array([1.0, 0.0, 0.0])
        xp = xg - np.dot(xg, self.ax) * self.ax
        if np.linalg.norm(xp) < 1e-9:
            xg = np.array([0.0, 1.0, 0.0])
            xp = xg - np.dot(xg, self.ax) * self.ax
        self.x_prime = _normalize(xp)

        # --- d_f: 馈源到反射面代表点的距离（口径边缘与中心的均值）---
        rim = reflector.rim(360) - self.W
        d_edge = float(np.mean(np.linalg.norm(rim, axis=1)))
        d_ctr = float(np.linalg.norm(reflector.center() - self.W))
        self.d_f = 0.5 * (d_edge + d_ctr)

        # --- (19) 窗口内选 w，再由 (15) 得 b、(13) 得 Δθ ---
        w_lo, w_hi = w_window(self.k, self.d_f, reflector.D)
        self.w_lo, self.w_hi = w_lo, w_hi
        self.w_rel = float(w_rel)
        self.w = w_lo * (w_hi / w_lo) ** self.w_rel if w is None else float(w)
        self.diag = check_13_19(self.k, self.d_f, reflector.D, self.w)
        if 'b' not in self.diag:
            raise ValueError(f"w={self.w:.6g} 落在 (19) 窗口 [{w_lo:.6g},{w_hi:.6g}] 之外或 (16) 不满足")
        self.b = self.diag['b']
        self.w0 = np.sqrt(2.0 * self.b / self.k)      # 腰半径
        self.dtheta = self.diag['delta_theta'] * float(dtheta_scale)   # (13)(可选加密)
        self.theta_gb = self.diag['theta_gb']

        # --- 格点 (10)(11) ---
        self.e1, self.e2 = _orthobasis(self.ax)
        self.lat = hex_lattice(self.theta_cover, self.dtheta)
        self.n_beams = len(self.lat)

        self.rim = reflector.rim(1440)
        self.beams = []
        self.residual = None

    # ---- 方向 / 基函数 ----
    def _dirs_of(self, ths):
        return np.array([_cone_dir(self.ax, t, p, self.e1, self.e2) for t, p in ths])

    def basis_ang(self, Q, dirs):
        """远区基函数 Ψ_n(Ω)：一束在方向 Ω 上的远场幅度比（相对轴上）。

        basis="exact"     Ψ = exp[-(tanγ/θ_gb)²]/cosγ   （由 (1) 的严格远区极限）
        basis="paraxial"  Ψ = exp[-(γ/θ_gb)²]
        两者在 γ→0 一致；大 γ 处 exact 明显更宽。
        """
        cosg = np.clip(Q @ np.asarray(dirs).T, -1.0, 1.0)
        gam = np.arccos(cosg)
        if self.basis == "paraxial":
            return np.exp(-(gam / self.theta_gb) ** 2)
        g = np.minimum(gam, 1.2)                       # γ>68.8° 时基函数已 ~0
        amp = np.exp(-(np.tan(g) / self.theta_gb) ** 2) / np.cos(g)
        return np.where(gam < 1.2, amp, 0.0)

    def _h_hat(self, d):
        h = np.cross(_normalize(d), self.x_prime)
        n = np.linalg.norm(h)
        return h / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])

    # ---- 分解 ----
    def build(self):
        dirs, ths = [], []
        for th, ph in self.lat:
            d = _cone_dir(self.ax, th, ph, self.e1, self.e2)
            O = self.refl.hit_ray(self.W, d)
            if O is None:
                continue
            lm = gs.LocalModelFit(self.refl.parab, O, self.rim, verbose=False)
            R = float(np.linalg.norm(O - self.W))
            to_l = np.stack([lm.xhat, lm.yhat, lm.zhat])
            zi_l = to_l @ _normalize(d)
            pol_l = to_l @ self._h_hat(d)
            beam = gs.GaussianBeam(self.b, R, zi_l, pol_ref=pol_l, k=self.k)
            self.beams.append(GBBeam(O=O, L=R, dir=_normalize(d), theta=float(th), lm=lm,
                                     beam=beam, inside=bool(self.refl.inside(O))))
            dirs.append(_normalize(d))
            ths.append(float(th))
        if not self.beams:
            raise RuntimeError("GBExpansionFarField: 没有束轴与反射面相交")
        self.dirs = np.asarray(dirs)
        self.ths = np.asarray(ths)
        self._solve_coefficients()

        inc = []
        for b in self.beams:
            dv = (b.O - self.W)
            dv = dv / np.linalg.norm(dv)
            inc.append(np.degrees(np.arccos(min(1.0, abs(dv @ b.lm.zhat)))))
        self.incidence_deg = np.asarray(inc, dtype=float)
        self.max_incidence_deg = float(np.max(self.incidence_deg))

        if self.verbose:
            n_out = sum(0 if b.inside else 1 for b in self.beams)
            dg = self.diag
            print(f"  [GB 分解(farfield)] 束数={len(self.beams)}, "
                  f"theta_cover={np.degrees(self.theta_cover):.2f}°, "
                  f"Δθ(13)={np.degrees(self.dtheta):.2f}°, theta_gb={np.degrees(self.theta_gb):.2f}°")
            print(f"             d_f={self.d_f:.4f} m, w={self.w*1e3:.2f} mm "
                  f"(窗口 [{self.w_lo*1e3:.1f}, {self.w_hi*1e3:.1f}] mm, w_rel={self.w_rel:g}), "
                  f"b(15)={self.b*1e3:.2f} mm, w0={self.w0*1e3:.2f} mm")
            print(f"             (16)={dg['eq16']} (17)={dg['eq17']} (18)={dg['eq18']} "
                  f"(19)={dg['eq19']};  反射点在口径内 {len(self.beams)-n_out}/{len(self.beams)}, "
                  f"远场匹配残差={self.residual:.3e}")
        return self

    def _solve_coefficients(self):
        """远场点匹配（chou2001 §II）:

            Σ_n H0_n (L_n + jb) e^{jkL_n} Ψ_n(γ_qn) = F(θ_q)

        匹配方向取 (10)(11) 的格点方向（N 个未知量、N 个方程，Tikhonov 正则），
        解得的就是反射点 O_n 处的入射场幅度 H0_n（与 near 版同量纲）。
        """
        Q = self.dirs
        thQ = np.arccos(np.clip(Q @ self.ax, -1.0, 1.0))
        L = np.array([b.L for b in self.beams])
        colfac = (L + 1j * self.b) * np.exp(1j * self.k * L)     # 远区 <-> O_n
        A = self.basis_ang(Q, self.dirs) * colfac[None, :]
        target = np.asarray(self.feed.far_field_amplitude(thQ), dtype=float)
        A_unit, t_unit = A.copy(), target.copy()
        if self.reg > 0:
            sc = np.linalg.norm(A) / np.sqrt(max(A.size, 1))
            A = np.vstack([A, np.sqrt(self.reg) * sc * np.eye(A.shape[1])])
            target = np.concatenate([target, np.zeros(A.shape[1])])
        sol, *_ = np.linalg.lstsq(A, target, rcond=None)
        self.sol = sol
        self.sing_vals = np.linalg.svd(A_unit, compute_uv=False)
        for b, h0 in zip(self.beams, sol):
            b.coef = complex(h0)
            b.beam.H0 = complex(h0) * b.beam.xi_dir.astype(complex)
        self.residual = float(np.linalg.norm(A_unit @ sol - t_unit)
                              / max(np.linalg.norm(t_unit), 1e-30))

    # ---- 自检（远场 + 面上）----
    def check_expansion(self, n_th=61, n_ph=24, verbose=True):
        """远场方向图上比较 Σ_n H0_n (L_n+jb)e^{jkL_n} Ψ_n 与馈源方向图；
        顺带报**反射面上**相对误差（这一项才是 GBMA 真正用到的量）。"""
        e1, e2 = _orthobasis(self.ax)
        ths = np.linspace(0.0, self.theta_cover, n_th)
        phs = np.linspace(0.0, 2.0 * _PI, n_ph, endpoint=False)
        Q = np.array([_cone_dir(self.ax, t, p, e1, e2) for t in ths for p in phs])
        L = np.array([b.L for b in self.beams])
        colfac = (L + 1j * self.b) * np.exp(1j * self.k * L)
        approx = self.basis_ang(Q, self.dirs) @ (colfac * self.sol)
        thr = np.repeat(ths, len(phs))
        tgt = np.asarray(self.feed.far_field_amplitude(thr), dtype=float)
        pk = np.max(np.abs(tgt))
        rel = np.abs(approx - tgt) / max(pk, 1e-30)

        rng = np.random.default_rng(0)
        rr = self.refl.a * np.sqrt(rng.random(400))
        pp = 2.0 * np.pi * rng.random(400)
        x = self.refl.ox + rr * np.cos(pp)
        y = self.refl.oy + rr * np.sin(pp)
        P = np.stack([x, y, self.refl.z(x, y)], axis=-1)
        tgt_s = self.feed_H(P)
        app_s = np.zeros_like(tgt_s)
        for b in self.beams:
            psi = b.beam.H(b.lm.to_local(P))
            app_s += psi @ np.stack([b.lm.xhat, b.lm.yhat, b.lm.zhat])
        num = np.linalg.norm(app_s - tgt_s, axis=-1)
        den = np.linalg.norm(tgt_s, axis=-1)
        rel_s = num / np.maximum(den, 1e-30)
        wgt = den ** 2 / np.sum(den ** 2)
        out = dict(ff_rel_max=float(rel.max()), ff_rel_rms=float(np.sqrt(np.mean(rel ** 2))),
                   surf_rel_w=float(np.sqrt(np.sum(wgt * rel_s ** 2))),
                   surf_rel_median=float(np.median(rel_s)))
        if verbose:
            print(f"  [分解检验(farfield)] 远场 |Δ|/|F|max = {out['ff_rel_max']:.3e} "
                  f"(RMS {out['ff_rel_rms']:.3e})")
            print(f"                       面上 光强加权 RMS = {out['surf_rel_w']:.3e} "
                  f"(中位 {out['surf_rel_median']:.3e})")
        return out

    # ---- 与 FeedGBExpansion 相同的馈源场模型（check_expansion 用）----
    def feed_H(self, P):
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
