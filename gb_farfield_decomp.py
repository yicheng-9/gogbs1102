# -*- coding: utf-8 -*-
"""
gb_farfield_decomp.py
=====================
馈源远场的**高斯波束分解**（Far-Field Gaussian-Beam Decomposition）——自包含实现。

方法来源
--------
H.-T. Chou, P. H. Pathak, R. J. Burkholder,
"Novel Gaussian beam method for the rapid analysis of large reflector antennas,"
*IEEE Trans. Antennas Propag.*, vol. 49, no. 6, pp. 883-893, Jun. 2001, §II.

核心思想
--------
一次反射的抛物面天线，其馈源照射可以**用一族旋转对称高斯波束（GB）近似展开**：

    F_feed(θ) ≈ Σ_n  C_n · F_GB( ∠(ŝ, ŝ_n) )                                    (★)

  · F_GB 是一支 GB 在远区的方向图（(3) 的远区极限）
  · ŝ_n 是第 n 支 GB 的方向，铺在覆盖反射面的**锥**里（六边形格点，Fig.4）
  · C_n 由最小二乘解出

于是每支 GB 可以**独立**地入射到反射面上、独立地做 GO 反射 + 边缘绕射（chou1997
的闭式解），最后相干叠加。这就是 GBMA 能把一次 PO 积分从"几小时"降到"几秒"的原因。

本文件**只做分解这一层**：给定馈源方向图 + 反射面几何，输出
  {方向 ŝ_n, 角距 Δθ, 共焦参数 b, 系数 C_n, 波束数 N}
并给出重构精度自检。拿去之后接你自己的逐束反射求解器即可。

参数选取链（原文 (13)-(19)，**N 是产物，不是输入**）
----------------------------------------------------
  (14)  w  = √( 2(d_f² + b²) / (k b) )                         光斑半径
  (15)  b  = k w²/4 − √( (k w²/4)² − d_f² )                     由 w 反解 b（小根）
  (16)  (k w²/4)² ≥ d_f²          ⇔  w ≥ 2√(d_f/k)              实根（下界 = 衍射极限）
  (17)  A₀ = π w² ≥ 4π d_f / k                                  光斑面积
  (18)  A₀ ≤ π D²/16              ⇔  w ≤ D/4                    闭式点位可用（上界）
  (19)  4 d_f/k ≤ w² ≤ D²/16                                    两条同时成立的窗口
  (13)  Δθ = √( 8 ln4 / (3 k b) )                               最近邻角距
        N  = 1 + 3R(R+1)   （覆盖锥半角 θ_max 所需的六边形格点数）

原文（p.885）：
  "Consequently, one may pick a convenient value of w which satisfies (19); this value
   of w in turn provides a choice for the value of b via (15) and hence also for Δθ
   via (13) thereby specifying each GB completely at launch. For more complex feed
   patterns, one can always decrease Δθ from its value in (13) while keeping the same
   b as in (15). In general, the conditions in (13) and (15) are sufficient for most
   types of simple feeds."

  ⟹ **w 在窗口内选**（本文用对数插值参数 w_rel ∈ [0,1]），b、Δθ、N 依次推出。
     "固定 b、减小 Δθ"是留给**更复杂馈源**的加密手段，不是常规做法。

两点必须知道的实测结论（不是原文，是本实现的经验）
--------------------------------------------------
1. **窗口下界不可用**。`w → w_min = 2√(d_f/k)` 时 `b → d_f`，反射面落进波束的瑞利
   长度内，而本方法的基函数 F_GB 是 GB 的**远区极限**（隐含 b ≪ r）。实测
   `w_rel < 0.25` 时方向图会崩（峰值偏差可达 +5 dB）。**推荐 w_rel ≈ 0.4**：
   此时 r/b ≈ 4.7，w/a ≈ 0.26，N(13) ≈ 37。
2. **必须检查 (18)**。`w > D/4` 会静默越过原文的闭式适用前提（chou1997 附录 B 的
   sin φ1se < 1/4）；本文件在 `b_from_w` 与 `decompose` 里都会显式报错/警告。

用法
----
    python gb_farfield_decomp.py            # 跑内置 demo + 自检
    python gb_farfield_decomp.py --export gb_set.npz

    # 作为库：
    from gb_farfield_decomp import Reflector, CosPowerFeed, decompose, reconstruct
    refl = Reflector(F=0.6, D=1.0)
    feed = CosPowerFeed.from_edge_taper(deg2rad(45.0), -12.0)
    dec  = decompose(refl, feed, freq=20e9, w_rel=0.40)
    amp  = reconstruct(dec, S)          # S: (M,3) 单位方向

怎么接到你自己的逐束反射求解器
------------------------------
`decompose()` 给出的 `dirs`(N,3)、`coef`(N,)、`b`、`k` 就是完整的波束集合。往后三步：

1. **求交**：每支 GB 的束腰放在**馈源相位中心**（本例即焦点），沿 `ŝ_n = dirs[n]`
   传播；与反射面求交得 `P_n`、`r_n = |P_n − 焦点|`、法向 `n̂`、局部的
   `w_inc(r_n) = w0√(1+(r_n/b)²)`、`R_inc(r_n) = r_n(1+(b/r_n)²)`。
2. **逐束的 PO / GO 反射**：入射场在反射点的**幅度**为

       H_n(P_n) = coef[n] · √(det Q(r_n)/det Q(0)) · e^{−jk r_n} / …       (19)(23)

   注意 `coef` 只是**幅度系数**；球面波前 `e^{−jk r_n}` 与横向高斯都得由各 GB 自己带
   （所以本文件输出的 `coef` 通常是**实数**——目标方向图是实幅度）。
3. **相干叠加**：把各束的远场（chou1997 (74)：`P̄_r·T(S_a) + P̄_d`）**先旋回全局基**，
   再乘 `exp(+jk ŝ·(P_n − O_0))` 统一相位参考点后相加——相位参考点不统一会引入
   线性相位斜坡（表现为主瓣偏移）。

本文件**不做**第 2、3 步，那是 chou1997 的闭式反射/绕射部分。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# =============================================================================
# 0. 常量
# =============================================================================
C0 = 299_792_458.0
LN4 = np.log(4.0)          # ln 4
_TWO_PI = 2.0 * np.pi


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


# =============================================================================
# 1. 反射面几何
# =============================================================================
@dataclass
class Reflector:
    """抛物面 z = (x²+y²)/(4F) 上的**投影口径圆盘**。

    参数
    ----
    F, D        : 焦距、投影口径直径（任意一致的长度单位，下文的 k 用同一单位）
    offset_x/y  : 口径圆心在口径面内的偏移（正馈时为 0）
    cone_margin : 覆盖锥半角相对"边缘最大角"的余量（**本实现的约定**，原文未给）

    派生量
    ------
    focus        : (0,0,F)
    feed_axis    : 焦点 → 口径圆心的单位方向
    cone_half    : θ_max，焦点看到的锥半角（含余量）
    d_f          : 焦点到边缘的平均距离，即 (14)(15)(19) 里的 d_f
    """

    F: float
    D: float
    offset_x: float = 0.0
    offset_y: float = 0.0
    cone_margin: float = 1.10
    n_rim: int = 720

    focus: np.ndarray = field(init=False, repr=False)
    feed_axis: np.ndarray = field(init=False, repr=False)
    cone_half: float = field(init=False)
    d_f: float = field(init=False)
    rim: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.focus = np.array([0.0, 0.0, float(self.F)])
        zc = (self.offset_x**2 + self.offset_y**2) / (4.0 * self.F)
        center = np.array([self.offset_x, self.offset_y, zc])
        self.feed_axis = _unit(center - self.focus)

        t = np.linspace(0.0, _TWO_PI, self.n_rim, endpoint=False)
        a = 0.5 * self.D
        x = self.offset_x + a * np.cos(t)
        y = self.offset_y + a * np.sin(t)
        self.rim = np.column_stack([x, y, (x * x + y * y) / (4.0 * self.F)])

        V = self.rim - self.focus
        R = np.linalg.norm(V, axis=1)
        cosang = np.clip(_unit(V) @ self.feed_axis, -1.0, 1.0)
        self.cone_half = float(np.arccos(cosang.min())) * self.cone_margin
        self.d_f = float(np.mean(R))

    @property
    def a_rim(self):
        return 0.5 * self.D


# =============================================================================
# 2. 馈源方向图（可插拔）
# =============================================================================
class CosPowerFeed:
    """经典的 cos^q(θ) 馈源模型：F(θ) = cos^q θ，θ 从馈源轴量起。

    用边缘照射角 θ_e 与边缘锥削 taper_dB 定 q：
        cos^q(θ_e) = 10^(taper_dB/20)   ⇒   q = (taper_dB/20) / log10(cos θ_e)
    """

    def __init__(self, q: float):
        self.q = float(q)

    @classmethod
    def from_edge_taper(cls, theta_e: float, taper_db: float) -> "CosPowerFeed":
        c = np.cos(theta_e)
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError("theta_e 必须在 (0, 90°) 内")
        return cls((taper_db / 20.0) / np.log10(c))

    def amplitude(self, theta):
        theta = np.asarray(theta, float)
        return np.cos(np.clip(theta, 0.0, 0.5 * np.pi)) ** self.q

    def __repr__(self):
        return f"CosPowerFeed(q={self.q:.3f})"


class GaussianFeed:
    """高斯馈源：远场方向图与一支 GB 同形，F(θ) = exp(−(k b_feed/2) sin²θ)。

    这是"最简单馈源"，分解几乎精确；用来验证代码链路。
    """

    def __init__(self, b_feed: float, k: float):
        self.b_feed = float(b_feed)
        self.k = float(k)

    def amplitude(self, theta):
        s2 = np.sin(np.asarray(theta, float)) ** 2
        return np.exp(-(self.k * self.b_feed / 2.0) * s2)

    def __repr__(self):
        return f"GaussianFeed(b_feed={self.b_feed:.4g})"


class CallableFeed:
    """把任意 `f(theta_rad) -> amplitude` 包装成馈源。"""

    def __init__(self, fn: Callable):
        self._fn = fn

    def amplitude(self, theta):
        return np.asarray(self._fn(np.asarray(theta, float)), float)


# =============================================================================
# 3. (13)-(19) 参数选取链
# =============================================================================
def spot_window(k: float, d_f: float, D: float):
    """(16)(17)(18)(19)：光斑半径 w 的允许窗口 (w_lo, w_hi)。

    w_lo = 2√(d_f/k)  —— (16)/(17) 的实根条件，等价于衍射极限（此时 b = d_f）
    w_hi = D/4        —— (18) 闭式点位可用的上界（chou1997 附录 B 的 sin φ1se<1/4）
    """
    return 2.0 * np.sqrt(d_f / k), D / 4.0


def w_from_rel(window, w_rel: float) -> float:
    """在 (19) 窗口内按对数插值选 w：w = w_lo·(w_hi/w_lo)^w_rel，0=下界，1=上界。"""
    lo, hi = window
    w_rel = float(w_rel)
    if not 0.0 <= w_rel <= 1.0:
        raise ValueError(f"w_rel={w_rel} 越界，必须在 [0,1]")
    return lo * (hi / lo) ** w_rel


def b_from_w(k: float, d_f: float, w: float) -> float:
    """(15) 的小根 b = k w²/4 − √((k w²/4)² − d_f²)，用有理化形式避免相消。

    同时校验 (16)：`(k w²/4)² ≥ d_f²`，即 w ≥ 2√(d_f/k)。
    """
    q = k * w * w / 4.0
    disc = q * q - d_f * d_f
    if disc < 0.0:
        raise ValueError(
            f"(16) 不满足：w={w:g} < 2√(d_f/k)={2.0*np.sqrt(d_f/k):g}"
            f"（式(15) 无实根，w 小于衍射极限）")
    return d_f * d_f / (q + np.sqrt(disc))


def check_window(k, d_f, D, w):
    """逐条核对 (16)(17)(18)(19)，返回诊断字典。"""
    lo, hi = spot_window(k, d_f, D)
    return dict(w=w, w_lo=lo, w_hi=hi,
                w_over_wlo=w / lo, w_over_whi=w / hi, w_over_D=w / D,
                eq16=bool(k * w * w / 4.0 >= d_f),
                eq18=bool(w <= hi), eq19=bool(lo <= w <= hi))


def delta_theta_from_b(k: float, b: float) -> float:
    """(13)：最近邻角距 Δθ = √(8 ln4 / (3 k b))。"""
    return float(np.sqrt(8.0 * LN4 / (3.0 * k * b)))


def delta_theta_for_count(cone_half: float, n_target: int) -> float:
    """反解：覆盖 cone_half 且点数 ≈ n_target 时所需的 Δθ。

    六边形格点 N = 1+3R(R+1)，故 R = (−1+√(1+4(N−1)/3))/2，Δθ = cone_half/R。
    对应原文允许的加密方式："固定 b、减小 Δθ"。
    """
    if n_target <= 1:
        return 2.0 * cone_half
    R = (-1.0 + np.sqrt(1.0 + 4.0 * (n_target - 1.0) / 3.0)) / 2.0
    return float(cone_half / max(R, 1.0))


# =============================================================================
# 4. 六边形格点（Fig.4 的排布：同心环，N = 1 + 3R(R+1)）
# =============================================================================
def hex_directions(cone_half: float, dtheta: float, tol: float = 1.05):
    """在"馈源轴为极轴"的球坐标里生成六边形格点方向，覆盖半角 cone_half。

    第 i 圈有 6i 个点，半径 i·Δθ，奇数圈错半格 —— 点数正好 1+3R(R+1)。
    """
    dirs = [np.array([0.0, 0.0, 1.0])]
    R = int(np.ceil(cone_half / dtheta))
    for i in range(1, R + 1):
        th = i * dtheta
        if th > cone_half * tol:
            break
        n = 6 * i
        phi = _TWO_PI * np.arange(n) / n + (np.pi / n if i % 2 else 0.0)
        st, ct = np.sin(th), np.cos(th)
        blk = np.column_stack([st * np.cos(phi), st * np.sin(phi), np.full(n, ct)])
        dirs.append(blk)
    return np.vstack(dirs)


def rotate_to_axis(v, axis):
    """把 +z 指向的点集旋转到 axis（罗德里格斯公式）。"""
    axis = _unit(axis)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, axis))
    if c > 1.0 - 1e-12:
        return np.array(v, copy=True)
    kv = np.cross(z, axis)
    kn = np.linalg.norm(kv)
    if kn < 1e-12:                      # 180°
        kv, ang = np.array([1.0, 0.0, 0.0]), np.pi
    else:
        kv, ang = kv / kn, np.arccos(c)
    s, cc = np.sin(ang), np.cos(ang)
    return (v * cc + np.cross(kv, v) * s
            + np.outer(v @ kv, kv) * (1.0 - cc))


def angle_between(dirs, ref):
    """dirs 与 ref 的夹角（dirs 可为 (N,3)，ref 可为 (M,3)）→ (M,N)。"""
    cosang = np.clip(np.asarray(ref) @ np.asarray(dirs).T, -1.0, 1.0)
    return np.arccos(cosang)


# =============================================================================
# 5. 基函数：一支 GB 的远区方向图
# =============================================================================
def gb_pattern(k: float, b: float, alpha, tilt: bool = True):
    """F_GB(α) = exp(−(k b / 2) sin²α) · [(1+cosα)/2]。

    由 (3) 的 Q 传播取远区极限得到：Q(0) = −j/b·I，ρ = z sinα，于是
        Re[−jk·½ρ²Q(z)] → −(k b/2) sin²α
    方括号里的 (1+cosα)/2 是 GB 自身的倾斜因子；`tilt=False` 可关掉，
    用于和"纯 cos^q"这类教科书馈源模型对比。
    """
    alpha = np.asarray(alpha, float)
    val = np.exp(-(k * b / 2.0) * np.sin(alpha) ** 2)
    return val * ((1.0 + np.cos(alpha)) / 2.0 if tilt else 1.0)


# =============================================================================
# 6. 主流程：分解
# =============================================================================
@dataclass
class Decomposition:
    """分解结果。拿去接逐束反射求解器即可。"""

    k: float
    b: float                      # (15) 基波束共焦参数
    dtheta: float                 # (13) 角距
    dirs: np.ndarray              # (N,3) 全局方向
    coef: np.ndarray              # (N,)  复系数 C_n
    cone_half: float
    window: tuple = (0.0, 0.0)
    info: dict = field(default_factory=dict)

    @property
    def N(self):
        return len(self.dirs)

    def __repr__(self):
        return (f"Decomposition(N={self.N}, b={self.b:.4g}, "
                f"dtheta={np.rad2deg(self.dtheta):.2f}°, "
                f"w={self.info.get('w', float('nan')):.4g})")


def decompose(refl: Reflector, feed, freq: float, w_rel: float = 0.40,
              tilt: bool = True, lsq_rcond: float = 1e-10,
              n_beams: int = None, verbose: bool = True) -> Decomposition:
    """馈源远场的 GB 分解。返回 Decomposition。

    步骤
    ----
    1. (19) 定窗口 → 由 w_rel 选 w（**越界立即报错**）
    2. (15) 由 w 解 b（含 (16) 校验）
    3. (13) 由 b 得 Δθ
    4. 六边形格点覆盖锥半角 → N（产物）
    5. 在"加密一倍的远场测试格点"上做 LSQ：
           Σ_n C_n · F_GB(∠(ŝ_t, ŝ_n)) = F_feed(θ_t)

    参数 n_beams
    -----------
    None（默认）= **严格按 (13)**，N 是产物。
    给定时 = 按原文允许的"固定 b、只减小 Δθ"加密（仅对更复杂的馈源方向图才有必要），
    并在 info['n_override'] 里标记，同时报告相对 N(13) 的倍数。
    """
    lam = C0 / float(freq)
    k = _TWO_PI / lam
    D, d_f = refl.D, refl.d_f

    # --- (19)(15)(13) ---
    win = spot_window(k, d_f, D)
    w = w_from_rel(win, w_rel)
    chk = check_window(k, d_f, D, w)
    if not chk["eq18"]:
        raise ValueError(
            f"(18) 不满足：w={w:g} > D/4={win[1]:g}；w_rel 请 ≤ 1")
    b = b_from_w(k, d_f, w)                       # (15)
    dtheta_13 = delta_theta_from_b(k, b)          # (13)
    n_13 = len(hex_directions(refl.cone_half, dtheta_13))

    dtheta = dtheta_13
    n_override = False
    if n_beams is not None and int(n_beams) > n_13:
        # 原文：固定 b，只减小 Δθ（仅复杂馈源才需要）
        dtheta = min(delta_theta_for_count(refl.cone_half, int(n_beams)), dtheta_13)
        n_override = True

    # --- 波束方向（N 是产物）---
    dirs_z = hex_directions(refl.cone_half, dtheta)
    dirs = rotate_to_axis(dirs_z, refl.feed_axis)
    N = len(dirs)

    # --- LSQ：在加密一倍的测试格点上匹配馈源远场方向图 ---
    test_z = hex_directions(refl.cone_half, dtheta / 2.0)
    test = rotate_to_axis(test_z, refl.feed_axis)
    A = gb_pattern(k, b, angle_between(dirs, test), tilt)      # (M,N)
    theta_t = np.arccos(np.clip(test @ refl.feed_axis, -1.0, 1.0))
    y = np.asarray(feed.amplitude(theta_t), float)             # (M,)
    coef, *_ = np.linalg.lstsq(A, y, rcond=lsq_rcond)

    resid = float(np.linalg.norm(A @ coef - y) / max(np.linalg.norm(y), 1e-300))
    amp_sum = float(np.sum(np.abs(coef)))
    net_sum = float(abs(np.sum(coef)))
    cond = amp_sum / max(net_sum, 1e-300)

    # --- 逐束的入射几何（供接反射求解器时用）---
    r_n = np.linalg.norm(
        _project_to_paraboloid(refl, dirs) - refl.focus[None, :], axis=1)
    r_over_b = float(np.median(r_n / b))

    info = dict(chk)
    info.update(lam=lam, r_over_b=r_over_b, N_test=len(test),
                lsq_rel=resid, sum_abs_over_net=cond,
                dtheta_13=dtheta_13, n_13=n_13,
                n_override=n_override,
                n_ratio=(N / n_13 if n_13 else 1.0))

    dec = Decomposition(
        k=k, b=b, dtheta=dtheta, dirs=dirs, coef=coef,
        cone_half=refl.cone_half, window=win, info=info)

    if verbose:
        _print_chain(refl, dec)
    return dec


def _project_to_paraboloid(refl: Reflector, dirs):
    """把从焦点出发的方向投到抛物面上（解 t 的一元二次，取**最小正根**）。

    注意：近根可能为负（射线先与抛物面的"另一叶"相交），必须取最小的正根，
    否则会得到一批 NaN。
    """
    F = refl.F
    s = _unit(dirs)
    W = refl.focus
    A = (s[:, 0] ** 2 + s[:, 1] ** 2) / (4.0 * F)
    B = 2.0 * (W[0] * s[:, 0] + W[1] * s[:, 1]) / (4.0 * F) - s[:, 2]
    Cq = (W[0] ** 2 + W[1] ** 2) / (4.0 * F) - W[2]

    t = np.full(len(s), np.nan)
    lin = np.abs(A) <= 1e-14
    # 线性分支（A≈0）
    with np.errstate(divide="ignore", invalid="ignore"):
        tl = -Cq / np.where(np.abs(B) > 1e-14, B, np.nan)
    t = np.where(lin & np.isfinite(tl) & (tl > 0), tl, t)
    # 二次分支：取最小正根
    quad = ~lin
    if np.any(quad):
        disc = B[quad] ** 2 - 4.0 * A[quad] * Cq
        ok = disc >= 0.0
        sq = np.sqrt(np.maximum(disc, 0.0))
        den = 2.0 * A[quad]
        t1 = (-B[quad] - sq) / den
        t2 = (-B[quad] + sq) / den
        cand = np.stack([t1, t2], axis=1)
        cand = np.where(np.isfinite(cand) & (cand > 0), cand, np.inf)
        best = np.min(cand, axis=1)
        best = np.where(np.isfinite(best), best, np.nan)
        tq = np.where(ok, best, np.nan)
        t[quad] = tq
    out = W[None, :] + t[:, None] * s
    return out


def _print_chain(refl: Reflector, dec: Decomposition):
    i = dec.info
    print(f"  反射面: F={refl.F:g}  D={refl.D:g}  F/D={refl.F/refl.D:g}  "
          f"偏移=({refl.offset_x:g},{refl.offset_y:g})")
    print(f"  锥半角 θ_max={np.rad2deg(refl.cone_half):.2f}°  "
          f"d_f={refl.d_f:.5g}  λ={i['lam']:.5g}  k={dec.k:.5g}")
    print(f"  (19) 窗口: w ∈ [{i['w_lo']:.5g}, {i['w_hi']:.5g}]  "
          f"(w/a ∈ [{i['w_lo']/refl.a_rim:.4f}, {i['w_hi']/refl.a_rim:.3f}])")
    print(f"  选定 w   = {i['w']:.5g}  → w/a={i['w']/refl.a_rim:.4f}, "
          f"w/w_lo={i['w_over_wlo']:.2f}, w/(D/4)={i['w_over_whi']:.2f}")
    print(f"  (16)={i['eq16']}  (18)={i['eq18']}  (19)={i['eq19']}")
    print(f"  (15) b   = {dec.b:.5g}     r̄/b = {i['r_over_b']:.2f}")
    print(f"  (13) Δθ  = {np.rad2deg(dec.dtheta):.3f}°"
          + ("" if not i["n_override"]
             else f"   〔已按原文'固定 b、减小 Δθ'加密："
                  f"N/N(13)={i['n_ratio']:.1f}，N(13)={i['n_13']}〕"))
    print(f"  N(产物)  = {dec.N}   测试格点 {i['N_test']}")
    print(f"  LSQ: rel.err={i['lsq_rel']:.3e}  Σ|C|/|ΣC|={i['sum_abs_over_net']:.3f}")


# =============================================================================
# 7. 重构与精度自检
# =============================================================================
def reconstruct(dec: Decomposition, S, tilt: bool = True):
    """在任意方向 S (M,3) 上重构馈源远场幅度 Σ_n C_n F_GB(∠(S, ŝ_n))。"""
    S = np.atleast_2d(np.asarray(S, float))
    A = gb_pattern(dec.k, dec.b, angle_between(dec.dirs, S), tilt)   # (M,N)
    return A @ dec.coef


def verify(dec: Decomposition, refl: Reflector, feed, n_theta: int = 400,
           tilt: bool = True, seed: int = 0) -> dict:
    """在与 LSQ 测试格点**不同**的随机方向上检验重构精度。

    用相对误差的三套统计：光强加权 RMS、中位数、95 分位。
    """
    rng = np.random.default_rng(seed)
    ct = np.cos(rng.uniform(0.0, refl.cone_half, n_theta))
    st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
    ph = rng.uniform(0.0, _TWO_PI, n_theta)
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ refl.feed_axis) > 0.95:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = _unit(e1 - (e1 @ refl.feed_axis) * refl.feed_axis)
    e2 = np.cross(refl.feed_axis, e1)
    S = (ct[:, None] * refl.feed_axis
         + st[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2))
    theta = np.arccos(np.clip(S @ refl.feed_axis, -1.0, 1.0))
    ref = np.asarray(feed.amplitude(theta), float)
    got = reconstruct(dec, S, tilt)
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12)
    w = ref**2 / max(np.sum(ref**2), 1e-300)
    return dict(rel_rms_weighted=float(np.sqrt(np.sum(w * rel**2))),
                rel_median=float(np.median(rel)),
                rel_p95=float(np.percentile(rel, 95)),
                rel_max=float(np.max(rel)),
                amp_ratio=float(np.max(np.abs(got)) / max(np.max(np.abs(ref)), 1e-300)))


# =============================================================================
# 8. 导出
# =============================================================================
def export_npz(path: str, dec: Decomposition):
    """把波束集合存成 npz，便于分享给别人接自己的反射求解器。"""
    np.savez(path,
             dirs=dec.dirs, coef=dec.coef, b=dec.b, dtheta=dec.dtheta,
             k=dec.k, cone_half=dec.cone_half, window=np.array(dec.window))
    return path


def export_csv(path: str, dec: Decomposition):
    """方向 + 复系数存成人类可读的 CSV。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("n,sx,sy,sz,re_C,im_C,abs_C,arg_C_deg\n")
        for n, (s, c) in enumerate(zip(dec.dirs, dec.coef)):
            f.write(f"{n},{s[0]:.9e},{s[1]:.9e},{s[2]:.9e},"
                    f"{c.real:.9e},{c.imag:.9e},{abs(c):.9e},"
                    f"{np.rad2deg(np.angle(c)):.6f}\n")
    return path


# =============================================================================
# 9. demo
# =============================================================================
def _sweep_w(refl, feed, freq, rows=9):
    """扫 w_rel，展示"N 是 (13)(15)(19) 的产物"以及窗口两端的退化。"""
    lam = C0 / freq
    k = _TWO_PI / lam
    print()
    print("=" * 78)
    print("① w_rel 扫描：(19) 窗口内的位置 → w → (15) b → (13) Δθ → N")
    print("=" * 78)
    print(f"{'w_rel':>6} {'w/a':>7} {'r̄/b':>6} {'b/d_f':>7} "
          f"{'Δθ(°)':>7} {'N':>4} {'LSQ rel':>10}  备注")
    for rel in np.linspace(0.0, 1.0, rows):
        try:
            d = decompose(refl, feed, freq, w_rel=rel, verbose=False)
        except ValueError as e:
            print(f"{rel:6.2f}  ---  {e}")
            continue
        note = ""
        if d.info["r_over_b"] < 2.0:
            note = "← 反射面已在波束近区，F_GB 的远区极限失效"
        elif d.N <= 19:
            note = "← 束太少（s/w>1），相邻波束有缝"
        print(f"{rel:6.2f} {d.info['w']/refl.a_rim:7.3f} "
              f"{d.info['r_over_b']:6.2f} {d.b/refl.d_f:7.3f} "
              f"{np.rad2deg(d.dtheta):7.2f} {d.N:4d} "
              f"{d.info['lsq_rel']:10.2e}  {note}")

    print()
    print("=" * 78)
    print("② 固定 b、只减小 Δθ（原文允许的加密）→ N 增大，LSQ 残差下降")
    print("=" * 78)
    print(f"{'N请求':>6} {'N实际':>6} {'Δθ(°)':>7} {'LSQ rel':>10} "
          f"{'重构 RMS':>10} {'Σ|C|/|ΣC|':>10}")
    for nb in (None, 37, 61, 91, 127, 169, 271, 397):
        try:
            d = decompose(refl, feed, freq, w_rel=0.40, n_beams=nb, verbose=False)
        except ValueError as e:
            print(f"{str(nb):>6}  --- {e}")
            continue
        ok = verify(d, refl, feed)
        print(f"{str(nb):>6} {d.N:6d} {np.rad2deg(d.dtheta):7.2f} "
              f"{d.info['lsq_rel']:10.2e} {ok['rel_rms_weighted']:10.2e} "
              f"{d.info['sum_abs_over_net']:10.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="馈源远场的高斯波束分解 (chou2001 §II)")
    ap.add_argument("--freq", type=float, default=20e9, help="频率 [Hz]，默认 20 GHz")
    ap.add_argument("--F", type=float, default=0.6, help="焦距 [m]")
    ap.add_argument("--D", type=float, default=1.0, help="投影口径直径 [m]")
    ap.add_argument("--ox", type=float, default=0.0, help="口径偏移 x [m]")
    ap.add_argument("--oy", type=float, default=0.0, help="口径偏移 y [m]")
    ap.add_argument("--edge-angle", type=float, default=45.0, help="边缘照射角 [deg]")
    ap.add_argument("--taper", type=float, default=-12.0, help="边缘锥削 [dB]")
    ap.add_argument("--w-rel", type=float, default=0.40, help="(19) 窗口内位置 [0,1]")
    ap.add_argument("--export", default=None, help="导出 .npz 路径")
    ap.add_argument("--export-csv", default=None, help="导出 .csv 路径")
    ap.add_argument("--no-sweep", action="store_true", help="跳过 w_rel 扫描")
    a = ap.parse_args(argv)

    refl = Reflector(F=a.F, D=a.D, offset_x=a.ox, offset_y=a.oy)
    feed = CosPowerFeed.from_edge_taper(np.deg2rad(a.edge_angle), a.taper)

    print("=" * 78)
    print("馈源远场高斯波束分解 —— chou2001 §II 式 (13)-(19)")
    print("=" * 78)
    print(f"  馈源模型: {feed}")
    dec = decompose(refl, feed, a.freq, w_rel=a.w_rel)
    ok = verify(dec, refl, feed)
    print(f"  重构自检（{400} 个随机方向，与 LSQ 测试格点无关）:")
    print(f"    光强加权 RMS 相对误差 = {ok['rel_rms_weighted']:.3e}")
    print(f"    中位 {ok['rel_median']:.3e}   95% {ok['rel_p95']:.3e}   "
          f"最大 {ok['rel_max']:.3e}")
    print(f"    峰值幅度比 = {ok['amp_ratio']:.5f}")

    # 用"高斯馈源"做一次精确性验证：此时基函数与目标同形，残差应极小
    print()
    print("-" * 78)
    print("链路自检：用一个与基函数同形的高斯馈源（b_feed = b），残差应≈0")
    print("-" * 78)
    gfeed = GaussianFeed(b_feed=dec.b, k=dec.k)
    dec2 = decompose(refl, gfeed, a.freq, w_rel=a.w_rel, verbose=False)
    ok2 = verify(dec2, refl, gfeed)
    print(f"  LSQ rel.err = {dec2.info['lsq_rel']:.3e}   "
          f"重构加权 RMS = {ok2['rel_rms_weighted']:.3e}")

    if not a.no_sweep:
        _sweep_w(refl, feed, a.freq)

    if a.export:
        print(f"\n  已导出波束集合: {export_npz(a.export, dec)}")
    if a.export_csv:
        print(f"  已导出 CSV: {export_csv(a.export_csv, dec)}")

    print()
    print("提醒（实测经验，非原文）：")
    print("  · w_rel < 0.25 时 r/b → 1，F_GB 的远区极限失效，方向图会崩 → 推荐 0.40")
    print("  · w > D/4 越过 (18)，本文件会直接报错（旧实现曾静默越界 +4 dB）")
    print("  · N 是 (13)+(19) 的产物；加密只在馈源复杂时按原文'固定 b、减小 Δθ'做")
    return dec


if __name__ == "__main__":
    main()
