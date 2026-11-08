"""
gb_single.py
============
Chou & Pathak (1997), Radio Sci. 32(4), 1319-1336:
"Uniform asymptotic solution for electromagnetic reflection and diffraction of
an arbitrary Gaussian beam by a smooth surface with an edge"

实现单个窄高斯波束照射反射面的闭式解, 即式 (74) 的 P^s:

        P^s(r) ~ P_r(r) * T(S_a) + P_d(r)                        ... (74)

    P_r  <- (68)            反射场 (二维驻定相位)
    P_d  <- (73)            边缘绕射场 (端点驻定相位 + 附录 B 的 phi1se)
    T    <- (65)            过渡函数 (误差函数)
    S_a  <- (69)(71)(72)

几何/相位系数 (2)(4)-(8)(15)-(20)(26)-(36)(40)-(49), 边缘参数 (41)-(49),
Jacobian 用 (53) (J = -AB*rho1), 附录 B 用 (B1)-(B23).
F(x',y') 用 (25) 的电场版本, 并把 1/R 取成 1 (即输出 r*E^s*exp(+jkr)).

论文未写死、本实现必须做出的三处约定:
  C1  (20)(28)-(32) 中 Q^i 取 O 点的 Q^i(0) (否则 f 不再是二次型)
  C2  局部抛物面 (2) 的 R1,R2 为负 (常规抛物面 z=(x^2+y^2)/4F 对应 R=-2F)
  C3  (68)(73) 的每个根号各自按 Im[sqrt(Z)] >= 0 取支, 不合并成 -jk*sqrt(a0a1-c^2)
另外正文 (B18) 的 (S+T) 按对称性取 (S-T); (B7) 缺失的 "+" 已补上.

两个不同的根号分支约定(混用会翻号):
    spim()  Im[sqrt(Z)] >= 0   —— (58)/(68)/(73) 的振幅因子
    srep()  Re[sqrt(Z)] >= 0   —— (71)(72) 里 sqrt(-jk g''_rho/2) 的符号判据

过渡区的数值稳定形式:
    (74) 的 T(S_a)*P_r 与 P_d 在 S_a -> 0 时各自发散(实测 1e13 / 1e18)再相消,
    领先阶正好等于 P_r*(1/2)[1+erf(S_a)]。代码同时给出 Ps(字面 74) 与
    Ps_stable(稳定式), 默认用 Ps_stable; 两者在 |S_a| >~ 2 处相差 < 0.5 dB。

实现中必须避开的坑(都是实测踩到过的):
    1) (36) 的 "+" 号: 写成负号 -> Im(f_s) 反号 -> e^{jk f_s} 随角度指数发散(+300 dB)
    2) 数值 PO 的观察方向 S 在**局部系**, 而面元坐标在**反射坐标系**, 必须旋转 S
       (否则主瓣会偏掉一个入射角 gamma, 看起来像"理论对了 PO 错了")
    3) (71)(72) 用 Re>0 分支, 不是 Im>=0 分支
    4) 局部模型的 (A,B,d) 拆分不唯一(结果与拆分无关), 本文按论文 (43a,b) 取 A,B,
       由 (43c) 定 d; x_o 的符号与 (46b) 自洽已做检查

长度单位一律取 1 个波长。验证结果见 REPORT.md。
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

# --------------------------------------------------------------------------- #
# 常数
# --------------------------------------------------------------------------- #
LAM = 1.0
K = 2.0 * np.pi / LAM
Z0 = 120.0 * np.pi
SQRT2PI = np.sqrt(2.0 * np.pi)
SQRTPI = np.sqrt(np.pi)


def spim(z):
    """sqrt(Z), 分支取 Im[sqrt(Z)] >= 0 (论文 (58) 之后的约定; 也是 (73) 里 sqrt(-2pi)=j*sqrt(2pi) 的依据)"""
    z = np.asarray(z, dtype=complex)
    r = np.sqrt(z)
    return np.where(r.imag >= 0.0, r, -r)


def srep(z):
    """sqrt(Z), 分支取 Re[sqrt(Z)] >= 0 —— (72) 对 sqrt(-jk g''_rho/2) 要求的是 Re>0,
    与 (58) 的 Im>=0 是两条不同的约定, 混用会让 (71) 的符号判据翻号。"""
    z = np.asarray(z, dtype=complex)
    r = np.sqrt(z)
    return np.where(np.real(r) < 0.0, -r, r)


def cbrt(z):
    """复数立方根主值 (B19)(B20)"""
    return np.power(np.asarray(z, dtype=complex), 1.0 / 3.0)


def _cubic_one_root(a2, a1, a0):
    """x^3 + a2 x^2 + a1 x + a0 = 0 的**一个**根（Cardano, 向量化, 复数）

    与 `phi1se_cubic` 里的 (B16)-(B23) 同一套公式; 单独抽出来供四次方程的
    Ferrari 分解式调用（那里需要解预解三次 z^3 + p z^2 + q z + r = 0）。
    """
    a2 = np.asarray(a2, dtype=complex)
    a1 = np.asarray(a1, dtype=complex)
    a0 = np.asarray(a0, dtype=complex)
    Q = (3.0 * a1 - a2 ** 2) / 9.0
    R = (9.0 * a2 * a1 - 27.0 * a0 - 2.0 * a2 ** 3) / 54.0
    disc = Q ** 3 + R ** 2
    Sst = cbrt(R + np.sqrt(np.asarray(disc, dtype=complex)))
    ok = np.abs(Sst) > 1e-300
    Tst = np.where(ok, -Q / np.where(ok, Sst, 1.0), 0.0)
    return Sst + Tst - a2 / 3.0


def _quartic_all_roots(qa, qb, qc, qd, qe):
    """qa s^4 + qb s^3 + qc s^2 + qd s + qe = 0 的 4 个根（Ferrari, 向量化, 复数）

    返回 shape = (4,) + broadcast(输入) 的数组; 系数 qa≈0 时按最低次处理不做,
    这类点由调用方的有效性判据筛掉。
    """
    qa = np.asarray(qa, dtype=complex)
    safe = np.where(np.abs(qa) > 1e-300, qa, 1.0)
    b = np.asarray(qb, dtype=complex) / safe
    c = np.asarray(qc, dtype=complex) / safe
    d = np.asarray(qd, dtype=complex) / safe
    e = np.asarray(qe, dtype=complex) / safe
    # 消去三次项: s = y - b/4  =>  y^4 + p y^2 + q y + r = 0
    p = c - 3.0 * b ** 2 / 8.0
    q = d - b * c / 2.0 + b ** 3 / 8.0
    r = e - b * d / 4.0 + b ** 2 * c / 16.0 - 3.0 * b ** 4 / 256.0
    # 分解 (y^2+A)^2 - (B y + C)^2, A 满足 A^3 - (p/2)A^2 - rA + (rp/2 - q^2/8) = 0
    A = _cubic_one_root(-p / 2.0, -r, r * p / 2.0 - q ** 2 / 8.0)
    B = np.sqrt(2.0 * A - p)
    okB = np.abs(B) > 1e-12
    C = np.where(okB, -q / (2.0 * np.where(okB, B, 1.0)), 0.0)
    out = []
    for bb, cst in ((B, A + C), (-B, A - C)):
        sq = np.sqrt(bb ** 2 - 4.0 * cst)
        out.append((-bb + sq) / 2.0 - b / 4.0)
        out.append((-bb - sq) / 2.0 - b / 4.0)
    return np.stack(out, axis=0)


# --------------------------------------------------------------------------- #
# 反射面
# --------------------------------------------------------------------------- #
class Paraboloid:
    """z = (x^2+y^2)/(4F), 沿 +z 张开, 投影口径半径 a 的圆盘."""

    def __init__(self, F, a):
        self.F = float(F)
        self.a = float(a)

    def z(self, x, y):
        return (x * x + y * y) / (4.0 * self.F)

    def normal(self, x, y):
        px = x / (2.0 * self.F)
        py = y / (2.0 * self.F)
        one = np.ones_like(px)
        n = np.stack([-px, -py, one], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def rim(self, n=720):
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = self.a * np.cos(t)
        y = self.a * np.sin(t)
        return np.stack([x, y, self.z(x, y)], axis=-1)


# --------------------------------------------------------------------------- #
# O 点局部模型 (2),(37)-(49)
# --------------------------------------------------------------------------- #
class LocalModel:
    """以反射点 O 为原点的局部抛物面 + 平面边缘的投影椭圆模型.

    局部系 (x,y,z): z 轴 = 表面外法向, x 轴 = 子午向(+x_r), y 轴 = 方位向,
        z = -0.5*(x^2/R1 + y^2/R2)                     ... (2), R1,R2 < 0 (C2)
    边缘位于平面 x2 = x_b 内 (37a)(38), 投影到 z=0 切平面后为椭圆 (41):
        (x-x_o)^2/A^2 + (y-y_o)^2/B^2 = d^2
    再用 (45)/(48)/(49) 的映射把椭圆内部写成 0 <= rho1 <= d, -pi <= phi1 <= pi.

    对"圆口径抛物面"这一情形 (38)(39) 给出解析系数:
        b11 = s = x_hat . z_r, b12 = 0, b13 = c = z_hat . z_r,
        x_b = dh = (a^2 - r0^2)/(4F)
    于是 (42)(43) 全部为闭式.
    """

    def __init__(self, parab, r0):
        F, a = parab.F, parab.a
        self.parab, self.F, self.a, self.r0 = parab, F, a, float(r0)

        p = self.r0 / (2.0 * F)
        q = 1.0 / np.sqrt(1.0 + p * p)

        self.O = np.array([self.r0, 0.0, self.r0 ** 2 / (4.0 * F)])
        self.zr = np.array([0.0, 0.0, 1.0])
        self.zhat = np.array([-p * q, 0.0, q])
        self.xhat = np.array([q, 0.0, p * q])
        self.yhat = np.array([0.0, 1.0, 0.0])

        km = (1.0 / (2.0 * F)) * q ** 3          # 子午向曲率
        ka = (1.0 / (2.0 * F)) * q               # 方位向曲率
        self.R1 = -1.0 / km                      # (2) 的主曲率半径 (负)
        self.R2 = -1.0 / ka

        self.s = p * q                           # b11
        self.cz = q                              # b13
        self.dh = (a * a - self.r0 * self.r0) / (4.0 * F)    # x_b

        self.A2 = -2.0 * self.R1 / self.cz       # (43a)
        self.B2 = -2.0 * self.R2 / self.cz       # (43b)
        self.A = float(np.sqrt(self.A2))
        self.B = float(np.sqrt(self.B2))
        self.xo = self.s * self.R1 / self.cz     # (42a)
        self.yo = 0.0                            # (42b)
        self.d2 = self.dh - self.s ** 2 * self.R1 / (2.0 * self.cz)      # (43c)
        self.d_imag = bool(self.d2 < 0.0)
        self.d = float(np.sqrt(abs(self.d2)))

        # (46) O 在 (rho, phi) 系中的位置; rho0=0 时按论文要求取 phi0=0
        self.rho0 = float(np.hypot(self.xo / self.A, self.yo / self.B))
        if self.rho0 > 1e-12:
            cph = -self.xo / (self.A * self.rho0)     # (46b)
            sph = -self.yo / (self.B * self.rho0)     # (46c)
            self.phi0 = float(np.arctan2(sph, cph))
        else:
            self.phi0 = 0.0

        self.A1 = self.A * np.cos(self.phi0)      # (49a)
        self.A2c = -self.A * np.sin(self.phi0)    # (49b)
        self.B1 = self.B * np.sin(self.phi0)      # (49c)
        self.B2c = self.B * np.cos(self.phi0)     # (49d)

        # 通用线性边缘映射 (48) 用它自己的 (A1,A2,B1,B2):
        #   x = x_o + [A1 cos(phi1) + A2 sin(phi1)] rho1
        #   y = y_o + [B1 cos(phi1) + B2 sin(phi1)] rho1
        # 对角情形 (49) 只是它的一个特例; 拟合椭圆模型 (LocalModelFit) 复用同一套公式.
        self.M = np.array([[self.A1, self.A2c], [self.B1, self.B2c]])
        self.Minv = np.linalg.inv(self.M)
        self.jac = float(abs(self.A2c * self.B1 - self.A1 * self.B2c))   # |det M| = A*B

    # ---- 通用边缘映射 ----
    def edge_map(self, phi):
        """返回 (Xr, Yr, Xr', Yr')，其中 Xr = dx/drho1, Yr = dy/drho1"""
        c, s = np.cos(phi), np.sin(phi)
        Xr = self.A1 * c + self.A2c * s
        Yr = self.B1 * c + self.B2c * s
        Xrp = -self.A1 * s + self.A2c * c
        Yrp = -self.B1 * s + self.B2c * c
        return Xr, Yr, Xrp, Yrp

    # ---- 映射 (45)/(48) ----
    def xy(self, rho, phi):
        Xr, Yr, _, _ = self.edge_map(phi)
        return self.xo + rho * Xr, self.yo + rho * Yr

    def xy_edge(self, phi):
        """rho1 = d 上的点 (44a)(44b), 即投影边缘"""
        return self.xy(self.d, phi)

    def rho_phi_of(self, x, y):
        """(45) 的反变换: 点 (x,y) 对应的 (rho1, phi1)

        注意 x,y 可以是复数: 高斯波束的二维驻定相位点本来就是复的
        (Q 为复数 -> (34)(35) 的 x_s,y_s 为复数), 这里保持解析延拓.
        rho1 取 Re>0 的分支(与实轴上 rho1>=0 的物理含义一致).
        """
        dx = np.asarray(x) - self.xo
        dy = np.asarray(y) - self.yo
        u = self.Minv[0, 0] * dx + self.Minv[0, 1] * dy
        v = self.Minv[1, 0] * dx + self.Minv[1, 1] * dy
        rho = np.sqrt(u * u + v * v)
        rho = np.where(np.real(rho) < 0.0, -rho, rho)
        phi = np.arctan2(v, u) - self.phi0
        return rho, phi

    def zloc(self, x, y):
        """局部抛物面 (2) 上的 z"""
        return -0.5 * (x * x / self.R1 + y * y / self.R2)

    def surf_nvec(self, x, y):
        """(15) 的非归一化法向 (其模在 (16)(17) 中与 dS 的因子正好相消)"""
        x = np.asarray(x, dtype=complex)
        y = np.asarray(y, dtype=complex)
        return (np.broadcast_to(self.zhat.astype(complex), x.shape + (3,))
                + x[..., None] * (self.xhat.astype(complex) / self.R1)
                + y[..., None] * (self.yhat.astype(complex) / self.R2))

    def to_local(self, P):
        P = np.asarray(P, dtype=float) - self.O
        return np.stack([P @ self.xhat, P @ self.yhat, P @ self.zhat], axis=-1)

    def to_global(self, Pl):
        Pl = np.asarray(Pl, dtype=float)
        return (self.O
                + Pl[..., 0:1] * self.xhat
                + Pl[..., 1:2] * self.yhat
                + Pl[..., 2:3] * self.zhat)

    def check(self, verbose=True):
        out = {}
        x0, y0 = self.xy(self.rho0, 0.0)     # (48) 在 O 点应给出 (0,0)
        out["O_mapping_err"] = float(np.hypot(x0, y0))
        rim = self.parab.rim(2000)
        rl = self.to_local(rim)
        val = (rl[:, 0] - self.xo) ** 2 / self.A2 + (rl[:, 1] - self.yo) ** 2 / self.B2
        out["rim_ellipse_max_rel_err"] = float(np.max(np.abs(val - self.d2)) / self.d2)
        if verbose:
            print("  [local model]")
            print(f"    R1={self.R1:9.3f}  R2={self.R2:9.3f}   (R1*R2>0: {self.R1*self.R2>0})")
            print(f"    A={self.A:8.4f}  B={self.B:8.4f}  d={self.d:8.4f}  "
                  f"x_o={self.xo:9.3f}  y_o={self.yo:6.3f}")
            print(f"    rho0={self.rho0:8.4f}  phi0={np.degrees(self.phi0):8.3f} deg  "
                  f"椭圆半轴: A*d={self.A*self.d:.3f}  B*d={self.B*self.d:.3f}")
            print(f"    自检: O 点映射误差={out['O_mapping_err']:.2e}   "
                  f"真实边缘 vs 椭圆(41) 最大相对误差={out['rim_ellipse_max_rel_err']:.3e}")
        return out


# --------------------------------------------------------------------------- #
# 通用局部模型: 边缘用实际 3D 边缘点投影后拟合椭圆(支持偏置口径 / 任意 O)
# --------------------------------------------------------------------------- #
def fit_ellipse_2d(px, py):
    """把二维点集拟合为椭圆, 返回 (center, M)。

    约定: (x, y) = center + M @ (cos t, sin t) 描出椭圆 (t 为参数)。
    做法: 归一化 -> Fitzgibbon 直接最小二乘椭圆拟合(代数距离) ->
          把圆锥曲线 (a,b,c,d,e,f) 化成中心 + 半轴矩阵。
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    x0 = 0.5 * (px.max() + px.min())
    y0 = 0.5 * (py.max() + py.min())
    s = max(np.max(np.abs(px - x0)), np.max(np.abs(py - y0)), 1e-30)
    X = (px - x0) / s
    Y = (py - y0) / s

    D1 = np.stack([X * X, X * Y, Y * Y], axis=-1)
    D2 = np.stack([X, Y, np.ones_like(X)], axis=-1)
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    Mtmp = S1 + S2 @ T
    Cmat = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
    evals, evecs = np.linalg.eig(np.linalg.solve(Cmat, Mtmp))
    best = None
    for i in range(3):
        a1 = np.real(evecs[:, i])
        a = np.concatenate([a1, T @ a1])
        if 4.0 * a[0] * a[2] - a[1] ** 2 > 0.0:      # 椭圆判据
            best = a
            break
    if best is None:                                  # 退化: 用最后一个
        a1 = np.real(evecs[:, 2])
        best = np.concatenate([a1, T @ a1])
    a = best
    A2 = np.array([[a[0], a[1] * 0.5], [a[1] * 0.5, a[2]]])
    b2 = np.array([a[3], a[4]])
    ctr = -0.5 * np.linalg.solve(A2, b2)
    k = float(ctr @ A2 @ ctr - a[5])
    Q = A2 / k
    ev, V = np.linalg.eigh(Q)
    ev = np.where(ev <= 0.0, 1e-30, ev)
    semi = 1.0 / np.sqrt(ev)
    Mmat = V @ np.diag(semi) * s
    ctr_full = np.array([x0, y0]) + s * ctr
    return ctr_full, Mmat


class LocalModelFit(LocalModel):
    """通用局部模型(GBMA 用): O 可以是抛物面上任意一点。

    - 局部系: ẑ = 外法向, x̂ = 子午向, ŷ = ẑ×x̂
    - 主曲率半径取负, 与 (2) 的符号约定一致
    - 边缘: 把实际 3D 边缘点投到 O 的切平面, 再拟合椭圆 (41) 的推广形式;
      于是 (A1,A2,B1,B2) 可以是**任意**线性映射(椭圆可旋转), 不要求与主方向对齐。
      取 d = 1, |J| = |det M| 作为 (53) 的通用形式。
    """

    def __init__(self, parab, O, rim, verbose=False):
        F = parab.F
        self.parab, self.F, self.a = parab, F, parab.a
        self.O = np.asarray(O, dtype=float)
        self.rim_points = np.asarray(rim, dtype=float)
        self.r0 = float(np.hypot(self.O[0], self.O[1]))

        px, py = self.O[0] / (2.0 * F), self.O[1] / (2.0 * F)
        nrm = np.array([-px, -py, 1.0])
        self.zhat = nrm / np.linalg.norm(nrm)
        rhat = (np.array([self.O[0], self.O[1], 0.0]) / self.r0
                if self.r0 > 1e-12 else np.array([1.0, 0.0, 0.0]))
        m = rhat - (rhat @ self.zhat) * self.zhat
        if np.linalg.norm(m) < 1e-12:
            m = np.array([1.0, 0.0, 0.0]) - self.zhat[0] * self.zhat
        self.xhat = m / np.linalg.norm(m)
        self.yhat = np.cross(self.zhat, self.xhat)

        p = self.r0 / (2.0 * F)
        q = 1.0 / np.sqrt(1.0 + p * p)
        self.R1 = -2.0 * F / q ** 3          # 子午向: 2F (1+p^2)^{3/2}
        self.R2 = -2.0 * F / q               # 方位向: 2F (1+p^2)^{1/2}

        P = self.rim_points - self.O
        ex = P @ self.xhat
        ey = P @ self.yhat
        ctr, Mmat = fit_ellipse_2d(ex, ey)
        self.xo, self.yo = float(ctr[0]), float(ctr[1])
        self.M = Mmat
        self.Minv = np.linalg.inv(self.M)
        self.jac = float(abs(np.linalg.det(self.M)))
        self.d = 1.0
        uv = self.Minv @ np.array([-self.xo, -self.yo])
        self.rho0 = float(np.hypot(uv[0], uv[1]))
        self.phi0 = float(np.arctan2(uv[1], uv[0])) if self.rho0 > 1e-12 else 0.0
        cs, sn = np.cos(self.phi0), np.sin(self.phi0)
        self.A1, self.B1 = self.M @ np.array([cs, sn])
        self.A2c, self.B2c = self.M @ np.array([-sn, cs])
        self.s = float(self.O[0] / (2.0 * F)) * q      # 仅作诊断
        self.cz = float(q)
        self.dh = float((parab.a ** 2 - self.r0 ** 2) / (4.0 * F))
        self.A = float(np.sqrt(abs(self.M[0, 0] ** 2 + self.M[0, 1] ** 2) + 0.0))
        self.B = float(np.sqrt(abs(self.M[1, 0] ** 2 + self.M[1, 1] ** 2) + 0.0))
        self.d_imag = False
        if verbose:
            self.check(verbose=True)

    def check(self, verbose=True):
        out = {}
        x0, y0 = self.xy(self.rho0, 0.0)
        out["O_mapping_err"] = float(np.hypot(x0, y0))
        P = self.rim_points - self.O
        ex = P @ self.xhat
        ey = P @ self.yhat
        u = self.Minv[0, 0] * (ex - self.xo) + self.Minv[0, 1] * (ey - self.yo)
        v = self.Minv[1, 0] * (ex - self.xo) + self.Minv[1, 1] * (ey - self.yo)
        rr = np.hypot(u, v)
        out["rim_fit_rel_err"] = float(np.max(np.abs(rr - 1.0)))
        if verbose:
            print("  [local model (fit)]")
            print(f"    R1={self.R1:9.3f}  R2={self.R2:9.3f}  |det M|={self.jac:8.4f}")
            print(f"    x_o={self.xo:9.3f}  y_o={self.yo:9.3f}  rho0={self.rho0:7.4f}  "
                  f"phi0={np.degrees(self.phi0):8.3f} deg")
            print(f"    自检: O 点映射误差={out['O_mapping_err']:.2e}   "
                  f"边缘点 |rho-1| 最大残差={out['rim_fit_rel_err']:.3e}")
        return out


# --------------------------------------------------------------------------- #
# 高斯波束 (1),(4)-(6)
# --------------------------------------------------------------------------- #
class GaussianBeam:
    """旋转对称高斯波束, 束坐标系 (x_i,y_i,z_i) 原点在 O, 沿 +z_i 传播.

        H^i(r_i) = H^i(O) sqrt(det Q(z_i)/det Q(0)) exp(-jk[z_i + 0.5 xi^T Q(z_i) xi])  (1)
        Q(z_i)^-1 = Q(0)^-1 + z_i * I                                                  (5)
        Q(0)^-1 = (L + j b) * I, 即束腰在 z_i = -L 处, w0 = sqrt(2b/k)                 (4)(6)

    只取旋转对称情形: Q = q(z)*I, (6) 的两个特征值相同.
    """

    def __init__(self, b, L, zhat_i, pol_ref, k=None):
        self.b = float(b)
        self.L = float(L)
        self.k = K if k is None else float(k)      # 波数(默认 2pi, 即长度为波长单位)
        z = np.asarray(zhat_i, dtype=float)
        self.zi_dir = z / np.linalg.norm(z)
        x = np.asarray(pol_ref, dtype=float)
        x = x - (x @ self.zi_dir) * self.zi_dir
        self.xi_dir = x / np.linalg.norm(x)
        self.yi_dir = np.cross(self.zi_dir, self.xi_dir)
        # (8): [r_i] = [c_ij][r_j]; c_ij 的第 i 行为 x_i,y_i,z_i 在局部系的坐标
        self.C = np.stack([self.xi_dir, self.yi_dir, self.zi_dir])
        self.H0 = self.xi_dir.astype(complex)     # H^i(O): 单位矢量, (1b) 保证垂直于 z_i
        self.w0 = float(np.sqrt(2.0 * self.b / self.k))

    def Q(self, zi):
        """(5) 标量情形"""
        return 1.0 / ((self.L + zi) + 1j * self.b)

    def det_ratio_sqrt(self, zi):
        """sqrt(det Q(z_i)/det Q(0))"""
        return (self.L + 1j * self.b) / ((self.L + zi) + 1j * self.b)

    def w(self, zi):
        return self.w0 * np.sqrt(1.0 + ((self.L + zi) / self.b) ** 2)

    def H(self, r_local):
        """入射磁场 H^i, r_local 为局部系坐标 (...,3) -- (1)(18)(19)(20)"""
        r = np.asarray(r_local, dtype=float)
        ri = r @ self.C.T
        xi, yi, zi = ri[..., 0], ri[..., 1], ri[..., 2]
        Qz = self.Q(zi)
        ph = -1j * self.k * (zi + 0.5 * Qz * (xi * xi + yi * yi))
        amp = self.det_ratio_sqrt(zi)
        return amp[..., None] * np.exp(ph)[..., None] * self.H0

    def H_amp_only(self, zi):
        """(19) 的 H^i_m: 只保留 det 比 (横向高斯已在 (20) 的 q 里)"""
        return self.det_ratio_sqrt(zi)[..., None] * self.H0

    def check(self, verbose=True):
        zz = np.array([-self.L, -0.5 * self.L, 0.0, 1.0, 5.0])
        amp = np.abs(self.det_ratio_sqrt(zz))
        ref = self.w(0.0) / self.w(zz)
        err = float(np.max(np.abs(amp - ref) / ref))
        if verbose:
            print(f"  [beam] b={self.b:.3f}  w0={self.w0:.4f}  L={self.L:.3f}  "
                  f"w(O)={self.w(0.0):.3f}   主轴幅度自检误差={err:.2e}")
        return {"beam_amp_err": err}


# --------------------------------------------------------------------------- #
# 闭式解 (74)
# --------------------------------------------------------------------------- #
class ChouPathakSingle:
    """单束 GB 的 P^s 闭式解. S 为观察方向 (局部系单位矢)."""

    def __init__(self, lm, beam, k=None, form_lo=0.7, form_hi=1.5, form_eps=1e-3,
                 edge_q="frozen", edge_crescent=False, crescent_order=1):
        self.lm = lm
        self.beam = beam
        self.k = beam.k if k is None else float(k)
        # 自动混合式的切换窗(|S_a|): 低于 form_lo 用稳定式, 高于 form_hi 用字面式
        self.form_lo = float(form_lo)
        self.form_hi = float(form_hi)
        # |S_a| 小于 form_eps 时字面式 T*P_r+P_d 的两项相消已不可信(纯舍入决定),
        # 直接用稳定式顶替: 该处两者本就差 <0.1%, 但字面式可能差 1e30 倍
        self.form_eps = float(form_eps)
        # edge_q: 'frozen' = 边缘相位沿用 Q(0) —— 论文约定 C1, **默认, 唯一推荐**
        #         'local'  = 边缘点改用局部 Q(z_i) 的实验性修正
        # 警告(实测): 'local' 会把边缘点被积函数**修正到几乎精确**
        #   (中心束误差 102 dB → 0.00 dB, 中间束 -2.22 → +1.98 dB),
        #   但 f_se 同时进入 S_a 与 T/P_d 的相消, 只改 f_se 会破坏统一定解的一致性 ——
        #   20 GHz/91 束实测峰 +369 dB、10-20° +112 dB. 而且它**修不了宽角偏低**:
        #   贴边束的边缘相位本来就精确(修正=0), 其宽角仍低 -1.2 ~ -2.7 dB.
        #   真正的宽角误差在端点项自身的领先阶求值(prefactor/g'_rho/g''_phi/单端点),
        #   要改需要重做端点积分, 不是两行补丁。
        self.edge_q = str(edge_q)
        # edge_crescent: 给直边截断因子 T 加**边缘曲率（月牙）修正**（REPORT A.24/A.25）
        #   来源: 用户提供的 slit_series.py（另一实现的 gbma_edge.slit_correction）。
        #   原理: 我们 (62) 的 T=½[1+erf(S_a)] 是"局部直边"结果; 真实曲边比直边
        #   少积一块月牙区 {d_r-κη²/2 <= ξ <= d_r}, 用 erf 的 Taylor 级数 + 复高斯矩
        #   闭式求和, 不做任何复根求解。默认**关**, 打开后 Ps 多返回 Ps_crescent。
        #   实测(A.25): 边界束 |T| 0.5000 → 0.4767, PO 反推 0.4547~0.4679。
        self.edge_crescent = bool(edge_crescent)
        self.crescent_order = int(crescent_order)
        self.Q0 = beam.Q(0.0)
        self.c = beam.C

    # ---------- (28)-(32) 的二次相位系数 ----------
    # 远场极限 (1/r -> 0) 下由 (26)+(20) 直接给出, 与 (28)-(32) 完全一致:
    #   a0 = 0.5[(z/r - c33)/R1 + c11^2 Q11 + c21^2 Q22 + 2 c11 c21 Q12]
    #   b0 = x/r - c31
    #   a1 = 0.5[(z/r - c33)/R2 + c12^2 Q11 + c22^2 Q22 + 2 c12 c22 Q12]
    #   b1 = y/r - c32
    #   c  = 0.5[c11 c12 Q11 + c21 c22 Q22 + (c11 c22 + c21 c12) Q12]
    def coeffs(self, S):
        lm, c = self.lm, self.c
        c11, c12, c13 = c[0]
        c21, c22, c23 = c[1]
        c31, c32, c33 = c[2]
        S = np.asarray(S, dtype=float)
        Sx, Sy, Sz = S[..., 0], S[..., 1], S[..., 2]
        Q11 = Q22 = self.Q0
        Q12 = 0.0
        a0 = 0.5 * ((Sz - c33) / lm.R1 + c11 ** 2 * Q11 + c21 ** 2 * Q22 + 2 * c11 * c21 * Q12)
        a1 = 0.5 * ((Sz - c33) / lm.R2 + c12 ** 2 * Q11 + c22 ** 2 * Q22 + 2 * c12 * c22 * Q12)
        b0 = Sx - c31
        b1 = Sy - c32
        cc = 0.5 * (c11 * c12 * Q11 + c21 * c22 * Q22 + (c11 * c22 + c21 * c12) * Q12)
        return a0, a1, b0, b1, cc

    # ---------- (33)-(36) 驻定相位点 ----------
    def stationary(self, S, c5=None):
        a0, a1, b0, b1, cc = self.coeffs(S) if c5 is None else c5
        den = a0 * a1 - cc ** 2
        xs = (a1 * b0 - b1 * cc) / (2.0 * den)                 # (34)
        ys = (a0 * b1 - b0 * cc) / (2.0 * den)                 # (35)
        # (36) 的 + 号: f(x_s,y_s) = -r + (a1 b0^2 + a0 b1^2 - 2 b0 b1 c)/(4(a0a1-c^2))
        # 若误写成负号, Im(f_s) 会反号, e^{jk f_s} 随角度指数发散(实测 +300 dB).
        fs = (a1 * b0 ** 2 + a0 * b1 ** 2 - 2.0 * b1 * b0 * cc) / (4.0 * den)
        return xs, ys, fs, den

    def _cache(self, S):
        """把**与 φ 无关**的量算一次: (a0,a1,b0,b1,cc, xs,ys,fs,den)

        Ps 内部有 phi1se 求解(含多次迭代/线搜索)、_edge_terms、gpp_rho 等多个子步骤都要
        用到这 9 个数；过去每处都重算一遍(实测每束 8 次 coeffs、4 次 stationary,
        占 Ps 的 24%)。现在统一算一次、按参数传下去。
        """
        c5 = self.coeffs(S)
        xs, ys, fs, den = self.stationary(S, c5)
        return c5 + (xs, ys, fs, den)


    # ---------- (25) F(x',y'), 电场版 ----------
    def fbar(self, x, y, S):
        """x,y 可为复数(边缘点 (x_se,y_se) 由复的 phi1se 给出, 见附录 B)

        ⚠ 坐标系: H0(bm.H_amp_only 的 H0)与 S 都是**局部系**分量(x̂,ŷ,ẑ 见 lm),
        所以曲面法向也必须换到局部系再叉乘。lm.surf_nvec() 返回的是**全局系**分量
        (它用全局的 lm.xhat/lm.yhat/lm.zhat 拼出来的), 直接用会把两个系混起来 ——
        REPORT A.19 记录了这个 bug: 偏馈 22.6° 入射时逐束幅度低估 24%,
        换到局部系后逐束 CF/PO 从 0.30~1.18 一律收敛到 0.995~0.996。
        """
        lm, bm = self.lm, self.beam
        x = np.asarray(x, dtype=complex)
        y = np.asarray(y, dtype=complex)
        S = np.broadcast_to(np.asarray(S, dtype=complex), x.shape + (3,))
        zp = lm.zloc(x, y)
        ri = np.stack([x, y, zp], axis=-1) @ bm.C.T
        Hm = bm.H_amp_only(ri[..., 2])                          # (19)
        nv = lm.surf_nvec(x, y)                                 # 全局分量
        nvl = np.stack([nv @ lm.xhat, nv @ lm.yhat, nv @ lm.zhat], axis=-1)  # -> 局部分量
        J = np.cross(nvl, Hm)                                   # (17) 的 J_eq dS'
        return (1j * self.k * Z0 / (2.0 * np.pi)) * np.cross(S, np.cross(S, J))   # (25), 1/R -> 1

    # ---------- 边缘曲率(月牙)修正: 几何 + 级数 (REPORT A.24/A.25) ----------
    def rim_nearest(self, xs, ys, iters=2):
        """(x_s,y_s) 到投影边缘椭圆 (rho1 = d) 的最近点 (实轴, 非复解)

        边缘是 (x,y) = (x_o,y_o) + d*[A1 cos φ + A2 sin φ, B1 cos φ + B2 sin φ]。
        最近点满足 h(φ) = (p - r)·r' = 0, 用牛顿解（初值取归一化径向方向）。
        实测：仅在径向初值上做 1 步牛顿，d_r 相对差 7e-5、kappa 5e-3，
        **S_ratio 与 8 步逐位相同** ⟹ 默认 2 步（96 ms → 见 _verify_crescent.py）。

        返回 dict: phi, d_r(口径内为正), er(朝外的单位矢), ep(切向), kappa(该点曲率)
        """
        lm = self.lm
        xs = np.real(np.asarray(xs, dtype=complex))
        ys = np.real(np.asarray(ys, dtype=complex))
        d = float(lm.d)
        dx, dy = xs - lm.xo, ys - lm.yo
        u = lm.Minv[0, 0] * dx + lm.Minv[0, 1] * dy
        v = lm.Minv[1, 0] * dx + lm.Minv[1, 1] * dy
        rho_n = np.hypot(u, v)                       # 归一化径向坐标, 边缘在 rho=d
        phi0 = np.arctan2(v, u) - lm.phi0            # 初值: 径向方向
        phi = phi0.copy()
        for _ in range(iters):
            c, s = np.cos(phi), np.sin(phi)
            Xr = lm.A1 * c + lm.A2c * s
            Yr = lm.B1 * c + lm.B2c * s
            Xrp = -lm.A1 * s + lm.A2c * c
            Yrp = -lm.B1 * s + lm.B2c * c
            px = xs - (lm.xo + d * Xr)
            py = ys - (lm.yo + d * Yr)
            tx, ty = d * Xrp, d * Yrp
            h = px * tx + py * ty
            hp = -(tx * tx + ty * ty) - d * (px * Xr + py * Yr)
            phi = phi - h / np.where(np.abs(hp) < 1e-300, 1e-300, hp)
        c, s = np.cos(phi), np.sin(phi)
        Xr = lm.A1 * c + lm.A2c * s
        Yr = lm.B1 * c + lm.B2c * s
        Xrp = -lm.A1 * s + lm.A2c * c
        Yrp = -lm.B1 * s + lm.B2c * c
        rx, ry = lm.xo + d * Xr, lm.yo + d * Yr
        # 法向: 与 (r - 椭圆中心) 同向者为"朝外"
        tx, ty = d * Xrp, d * Yrp
        tn = np.hypot(tx, ty)
        nx, ny = ty / tn, -tx / tn
        flip = (nx * (rx - lm.xo) + ny * (ry - lm.yo)) < 0.0
        nx = np.where(flip, -nx, nx)
        ny = np.where(flip, -ny, ny)
        dist = np.hypot(rx - xs, ry - ys)
        # 曲率: κ = |det M| / (d |r'|^3)  (见 REPORT A.25.1)
        kappa = lm.jac / (d * tn ** 3)
        return {"phi": phi, "d_r": np.where(rho_n <= d, dist, -dist),
                "er": np.stack([nx, ny], axis=-1),
                "ep": np.stack([-ny, nx], axis=-1),
                "kappa": kappa, "dist": dist}

    # ---------- 只算反射本体（论文 L750-753 的"边缘不存在时的反射场"）----------
    def Pr_only(self, S, cache=None):
        """只返回 P_r，**不碰** (B7)/端点项/过渡函数 —— 供 form="go" 的快速通道

        P_r 只依赖 (a0,a1,cc,xs,ys,fs,den) 与鞍点处的矢量幅度，因此省掉
        phi1se 复根求解、_edge_terms、gpp、erf 等全部边缘机件（实测快 3~4×）。
        """
        S = np.atleast_2d(np.asarray(S, dtype=float))
        c5 = self._cache(S) if cache is None else cache
        a0, a1, b0, b1, cc, xs, ys, fs, den = c5
        with np.errstate(all="ignore"):
            Fs = self.fbar(xs, ys, S)
            w1 = spim(-1j * self.k * a1)
            w2 = spim(-1j * self.k * (den / a1))
            Pr = (-np.pi / (w1 * w2))[..., None] * Fs * np.exp(1j * self.k * fs)[..., None]
        bad = ~np.isfinite(Pr).all(axis=-1)
        if np.any(bad):
            Pr = np.where(bad[..., None], 0.0, Pr)
        return Pr

    def crescent_ratio(self, S, cache=None, order=None):
        """月牙(曲边)修正 S_ratio: 真实曲边比"切线直边"少积的那块

        在**鞍点处**展开 f, 一次项恒为 0 ⟹ u_r = u_phi = 0 ⟹ Z = sqrt(c_r)·d_r。
        符号约定: c = +jk*a（`coeffs()` 的 a 与物理二次相位反号, 由 Re c > 0
        = 高斯衰减唯一确定; 校验: 1/sqrt(Re c_phi) 必须等于束斑半径 w）。
        交叉项用 Schur 补吸收: a_eff = a_ξξ - a_ξη²/a_ηη。

        返回 dict: S_ratio, Z, T_straight(=½[1+erf Z]), d_r, kappa, ok
        """
        lm, k = self.lm, self.k
        if cache is None:
            cache = self._cache(S)
        a0, a1, b0, b1, cc, xs, ys = (cache[0], cache[1], cache[2], cache[3],
                                      cache[4], cache[5], cache[6])
        order = self.crescent_order if order is None else int(order)
        _shp = np.shape(np.real(np.asarray(xs)))
        zero = np.zeros(np.broadcast_shapes(_shp, np.shape(np.real(np.asarray(ys)))),
                        dtype=float)
        out0 = {"S_ratio": zero.astype(complex), "Z": zero.astype(complex),
                "T_straight": np.full_like(zero, 0.5), "d_r": zero,
                "kappa": zero, "ok": np.zeros(np.shape(zero), dtype=bool)}
        if (not np.isfinite(lm.d)) or lm.d <= 0.0 or bool(getattr(lm, "d_imag", False)):
            return out0

        gm = self.rim_nearest(xs, ys)
        er, ep = gm["er"], gm["ep"]
        a0 = np.asarray(a0, dtype=complex) + 0.0 * er[..., 0]
        a1 = np.asarray(a1, dtype=complex) + 0.0 * er[..., 0]
        cc = np.asarray(cc, dtype=complex) + 0.0 * er[..., 0]
        ex, ey = er[..., 0], er[..., 1]
        px_, py_ = ep[..., 0], ep[..., 1]
        a_rr = a0 * ex * ex + 2.0 * cc * ex * ey + a1 * ey * ey
        a_rp = a0 * ex * px_ + cc * (ex * py_ + ey * px_) + a1 * ey * py_
        a_pp = a0 * px_ * px_ + 2.0 * cc * px_ * py_ + a1 * py_ * py_
        with np.errstate(all="ignore"):
            a_eff = a_rr - a_rp ** 2 / np.where(np.abs(a_pp) < 1e-300, 1e-300, a_pp)
            c_r = 1j * k * a_eff                    # = -jk*(物理 a), 见 (A.25.1)
            c_p = 1j * k * a_pp
            ok = (np.real(c_r) > 0.0) & (np.real(c_p) > 0.0)
            ok &= np.isfinite(c_r) & np.isfinite(c_p)
            sq = srep(np.where(ok, c_r, 1.0 + 0.0j))          # Re sqrt >= 0
            Z = np.where(ok, sq * gm["d_r"], 0.0)
            T_st = 0.5 * (1.0 + np.where(ok, erf(Z), 0.0))
            s2 = 1.0 / np.where(ok, 2.0 * c_p, 1.0 + 0.0j)
            lam = np.where(ok, sq * gm["kappa"] / 2.0, 0.0)
            # m1 = 0 ⟹ M_2n = (2n-1)!! s2^n
            acc = lam * s2                                     # n=1 项
            if order >= 2:
                M4 = 3.0 * s2 ** 2
                acc = acc + lam ** 2 * (2.0 * Z) * M4 / 2.0    # H_1(Z) = 2Z
            if order >= 3:
                M6 = 15.0 * s2 ** 3
                acc = acc + lam ** 3 * (4.0 * Z ** 2 - 2.0) * M6 / 6.0
            S_ratio = np.where(ok, np.exp(-Z * Z) * acc / SQRTPI, 0.0)
            S_ratio = np.where(np.isfinite(S_ratio), S_ratio, 0.0)
        return {"S_ratio": S_ratio, "Z": Z, "T_straight": T_st, "d_r": gm["d_r"],
                "kappa": gm["kappa"], "ok": ok}

    # ---------- 附录 B: rho1=d 处的 phi1se ----------
    def _edge_terms(self, S, phi, cache=None):
        lm = self.lm
        if cache is None:
            cache = self._cache(S)
        a0, a1, b0, b1, cc, xs, ys, fs, den = cache
        Xr, Yr, Xrp, Yrp = lm.edge_map(phi)
        x = lm.xo + lm.d * Xr
        y = lm.yo + lm.d * Yr
        fx = -2.0 * a0 * (x - xs) - 2.0 * cc * (y - ys)
        fy = -2.0 * a1 * (y - ys) - 2.0 * cc * (x - xs)
        # D 正比于 dg/dphi1 = fx*x_phi + fy*y_phi, 这里去掉公共因子 d
        D = Xrp * fx + Yrp * fy
        dfx = -2.0 * a0 * (lm.d * Xrp) - 2.0 * cc * (lm.d * Yrp)
        dfy = -2.0 * a1 * (lm.d * Yrp) - 2.0 * cc * (lm.d * Xrp)
        # dD/dphi1 = Xr''*fx + Xr'*dfx + Yr''*fy + Yr'*dfy, 而 Xr'' = -Xr, Yr'' = -Yr
        dD = -Xr * fx - Yr * fy + Xrp * dfx + Yrp * dfy
        return x, y, fx, fy, D, dD

    def phi1se_newton(self, S, phi_start=None, iters=80, tol=1e-14,
                      max_step=0.3, verbose=False, cache=None):
        """数值解 (B6)(B7): g'_phi(phi1se)|_{rho1=d} = 0 (复 Newton + 阻尼线搜索 + 兜底)

        工程健壮性(GBMA 里每束都要解一次, 各种几何都可能遇到):
          * 步长上限 max_step, 防止首次迭代跳飞
          * 每次迭代做 1,1/2,1/4,1/8,1/16 步长的线搜索, 只接受让 |D| 下降的步
          * dD ~ 0 的**退化情形**(例如束轴正好打在反射面对称中心, g 与 phi 无关):
            D 恒为 0, 任何 phi 都是解, 此时保持 phi = 0(对称解), 不迭代
          * 迭代后仍非有限/未收敛的方向退回 phi = 0 (即"朝最近边缘"的方向)
        """
        S = np.asarray(S, dtype=float)
        if cache is None:
            cache = self._cache(S)
        NN = S.shape[:-1]
        if phi_start is None:
            phi = np.zeros(NN, dtype=complex)
        else:
            phi = np.array(np.broadcast_to(np.asarray(phi_start, dtype=complex), NN),
                           dtype=complex)
        phi = np.where(np.isfinite(phi), phi, 0.0)
        with np.errstate(all="ignore"):
            for _ in range(iters):
                x, y, fx, fy, D, dD = self._edge_terms(S, phi, cache)
                Dabs = np.abs(D)
                good = np.isfinite(Dabs) & np.isfinite(dD) & (np.abs(dD) > 1e-300)
                step = np.where(good, D / np.where(good, dD, 1.0), 0.0)
                mag = np.abs(step)
                step = step * np.where(mag > max_step, max_step / np.maximum(mag, 1e-300), 1.0)
                best_phi, best_D = phi, Dabs
                for ls in range(5):
                    trial = phi - step * (0.5 ** ls)
                    xt, yt, fxt, fyt, Dt, _ = self._edge_terms(S, trial, cache)
                    Dt = np.abs(Dt)
                    better = np.isfinite(Dt) & (Dt < best_D)
                    best_phi = np.where(better, trial, best_phi)
                    best_D = np.where(better, Dt, best_D)
                moved = np.abs(best_phi - phi)
                phi = best_phi
                if np.all(moved < tol):
                    break
        bad = ~np.isfinite(phi) | (np.abs(np.imag(phi)) > 3.0) | (np.abs(np.real(phi)) > np.pi)
        if verbose:
            print(f"    [phi1se] 兜底方向数 {int(np.sum(bad))}/{phi.size}")
        phi = np.where(bad, 0.0 + 0.0j, phi)
        return phi

    def phi1se_cubic(self, S, cache=None):
        """论文附录 B 的闭式 (B14)-(B23): s = sin(phi1se) 满足三次方程"""
        lm = self.lm
        if cache is None:
            cache = self._cache(S)
        a0, a1, b0, b1, cc, xs, ys, fs, den = cache
        X = lm.xo - xs
        Y = lm.yo - ys
        A1, A2, B1, B2 = lm.A1, lm.A2c, lm.B1, lm.B2c
        d = lm.d
        H1 = 4.0 * d * (a0 * A1 * A2 + a1 * B1 * B2 + cc * (A2 * B1 + B2 * A1))       # (B8)
        H2 = 2.0 * d * (a0 * (A1 ** 2 - A2 ** 2) + a1 * (B1 ** 2 - B2 ** 2)
                        + 2.0 * cc * (A1 * B1 - A2 * B2))                             # (B9)
        H3 = 2.0 * (a0 * A1 * X + a1 * B1 * Y + cc * (A1 * Y + B1 * X))               # (B10)
        H4 = -2.0 * (a0 * A2 * X + a1 * B2 * Y + cc * (A2 * Y + B2 * X))              # (B11)
        H5 = -0.5 * H1                                                                # (B12)
        al1 = (H4 - 2.0 * H1) / H2                                                    # (B14)
        al2 = -2.0 * (H2 + H3) / H2
        al3 = -2.0 * (H4 + H5) / H2
        Q = (3.0 * al2 - al1 ** 2) / 9.0                                             # (B21)
        R = (9.0 * al1 * al2 - 27.0 * al3 - 2.0 * al1 ** 3) / 54.0                    # (B22)
        disc = Q ** 3 + R ** 2
        Sst = cbrt(R + np.sqrt(disc))                                                # (B19)
        Tst = -Q / Sst                                                               # (B23) S*T = -Q
        z1 = Sst + Tst - al1 / 3.0                                                    # (B16)
        z2 = -0.5 * (Sst + Tst) - al1 / 3.0 + 0.5j * np.sqrt(3.0) * (Sst - Tst)       # (B17)
        z3 = -0.5 * (Sst + Tst) - al1 / 3.0 - 0.5j * np.sqrt(3.0) * (Sst - Tst)       # (B18) 修正 (S+T)->(S-T)
        cand = np.stack([z1, z2, z3], axis=0)
        idx = np.argmin(np.abs(cand), axis=0)          # 取 |sin(phi1se)| 最小的根
        z = np.take_along_axis(cand, idx[None, ...], axis=0)[0]
        return np.arcsin(np.asarray(z, dtype=complex))

    def phi1se_quartic(self, S, tol=1e-8, cache=None):
        """附录 B (B7) 的**精确**闭式解: 不取 (B13) 小角近似, 直接解四次方程

        论文 L1051-1053 原话: "φ₁ₛe in (B7) can be found in closed form by replacing
        cos φ₁ₛe with √(1−sin²φ₁ₛe) and then solving a quartic equation for sin φ₁ₛe at the
        expense of some tedious algebra. However, a significant simplification ... if one
        approximates √(1−sin²φ₁ₛe) for cos φ₁ₛe as [1−sin²/2] (B13)."
        即: **四次**是论文说的 exact 路线, **三次**(`phi1se_cubic`) 是 (B13) 的近似产物,
        后者要求 sin φ₁ₛe < 1/4. 这里实现四次, 从而**没有那个前提**.

        (B7): H1 s² + H2 s c + H3 s + H4 c + H5 = 0
        对 c 是线性的 ⟹ c = −P/Q, P = H1 s² + H3 s + H5, Q = H2 s + H4;
        再代入 c² = 1 − s² ⟹ P² − (1−s²)Q² = 0 —— 关于 s 的四次方程:
            (H1²+H2²)s⁴ + 2(H1H3+H2H4)s³ + (H3²+2H1H5−H2²+H4²)s²
            + 2(H3H5−H2H4)s + (H5²−H4²) = 0
        平方会引入 c=+P/Q 的**伪根**, 用 |s²+c²−1| < tol 剔除, 再取 |s| 最小的那个
        (与三次解取 |sin| 最小根的口径一致 ⟹ 落在同一支上)。

        返回 (phi, ok); ok=False 表示该点四次解不可用(需回退)。
        """
        lm = self.lm
        if cache is None:
            cache = self._cache(S)
        a0, a1, b0, b1, cc, xs, ys, fs, den = cache
        X = lm.xo - xs
        Y = lm.yo - ys
        A1, A2, B1, B2 = lm.A1, lm.A2c, lm.B1, lm.B2c
        d = lm.d
        H1 = 4.0 * d * (a0 * A1 * A2 + a1 * B1 * B2 + cc * (A2 * B1 + B2 * A1))       # (B8)
        H2 = 2.0 * d * (a0 * (A1 ** 2 - A2 ** 2) + a1 * (B1 ** 2 - B2 ** 2)
                        + 2.0 * cc * (A1 * B1 - A2 * B2))                             # (B9)
        H3 = 2.0 * (a0 * A1 * X + a1 * B1 * Y + cc * (A1 * Y + B1 * X))               # (B10)
        H4 = -2.0 * (a0 * A2 * X + a1 * B2 * Y + cc * (A2 * Y + B2 * X))              # (B11)
        H5 = -0.5 * H1                                                                # (B12)
        with np.errstate(all="ignore"):
            qa = H1 ** 2 + H2 ** 2
            qb = 2.0 * (H1 * H3 + H2 * H4)
            qc = H3 ** 2 + 2.0 * H1 * H5 - H2 ** 2 + H4 ** 2
            qd = 2.0 * (H3 * H5 - H2 * H4)
            qe = H5 ** 2 - H4 ** 2
            # 归一化(防止四次项系数是舍入噪声时把整条方程放大到 1e29 量级)
            scale = np.maximum(np.maximum(np.maximum(np.abs(qa), np.abs(qb)),
                                          np.maximum(np.abs(qc), np.abs(qd))), np.abs(qe))
            scale = np.where(scale > 0.0, scale, 1.0)
            # 四阶系数相对过小 ⟹ 方程退化(例: 束轴打在对称中心时 H1=H2=0),
            # 这类点直接判无效, 由上层回退(退化情形任何 phi 都是解, 取 phi=0 或牛顿)
            deg_ok = np.abs(qa) > 1e-8 * scale
            roots = _quartic_all_roots(qa / scale, qb / scale, qc / scale,
                                       qd / scale, qe / scale)
            P = H1 * roots ** 2 + H3 * roots + H5
            Q = H2 * roots + H4
            okq = np.abs(Q) > 1e-300
            C = np.where(okq, -P / np.where(okq, Q, 1.0), np.nan)
            err = np.abs(roots ** 2 + C ** 2 - 1.0)          # 剔伪根 (c=+P/Q 那些)
            ok = np.isfinite(err) & (err < tol)
            # s -> phi: arcsin 只给 principal 支(c>0), 用 c=-P/Q 判断该取哪一支
            phi0 = np.arcsin(np.asarray(roots, dtype=complex))
            phi1 = np.pi - phi0                              # sin 相同、cos 反号
            phi0 = np.where(np.abs(np.cos(phi1) - C) < np.abs(np.cos(phi0) - C), phi1, phi0)
            # 把 Re(phi) 折回 (-pi, pi]
            phi0 = np.where(np.real(phi0) > np.pi, phi0 - 2.0 * np.pi, phi0)
            phi0 = np.where(np.real(phi0) < -np.pi, phi0 + 2.0 * np.pi, phi0)
            ok = ok & deg_ok & np.isfinite(phi0)
            # (B7) 有多个真根(它们都精确满足方程), 但**只有一支**对应"离 O 最近的那段
            # 边缘", 选错根会让 (71)(72) 的过渡分支翻号 → P_d 变成垃圾(实测 +200 dB)。
            # 三次解(B13 近似)虽然精度只有 1e-5 量级, 但它落在同一支上, 正好当"选根器":
            # 四次解负责精度, 三次解负责分支。
            pc = np.asarray(self.phi1se_cubic(S, cache), dtype=complex)
            # 退化根: H1=0 ⟹ H5=0 ⟹ phi=0 本身满足 (B7)(此时任何 phi 都满足, 取对称解)。
            # 这类点上四个四次根里往往没有 phi=0(Q -> 0 使 c=-P/Q 无意义), 必须显式补上,
            # 否则会选到 |Im phi|~3 的远根(P_d 放大到 1e58)。
            hscale = np.maximum(np.maximum(np.maximum(np.abs(H1), np.abs(H2)),
                                           np.maximum(np.abs(H3), np.abs(H4))), np.abs(H5))
            hscale = np.where(hscale > 0.0, hscale, 1.0)
            zero_ok = np.abs(H5) <= 1e-10 * hscale
            zcand = np.zeros_like(phi0[:1])
            phi0 = np.concatenate([phi0, zcand], axis=0)
            ok = np.concatenate([ok, np.broadcast_to(zero_ok[None, ...], zcand.shape)], axis=0)
            score = np.where(ok, np.abs(phi0 - pc[None, ...]), np.inf)
            j = np.argmin(score, axis=0)
            phi = np.take_along_axis(phi0, j[None, ...], axis=0)[0]
            ok = np.take_along_axis(ok, j[None, ...], axis=0)[0]
        return phi, ok

    # ---------- 统一入口(便于对比几种 phi1se 解法) ----------
    def phi1se(self, S, mode="newton", cache=None):
        """按 mode 求边缘点的 phi1se; 返回 phi(复数, 形状 = S.shape[:-1])

        mode
        ----
        "newton"       : 阻尼牛顿解精确 (B7)               —— **默认**, 最稳
        "cubic"        : 三次解(B13近似)做起初值 + 牛顿    —— 等价精度, 省几次迭代
        "cubic_pure"   : 纯三次闭式, 无牛顿                —— 有 (B13) 前提, 快但可能错
        "quartic_pure" : 纯四次闭式(无 B13、无牛顿)        —— 坏点退回 phi=0
        "closed"       : 四次闭式; 坏点依次回退 三次 → 牛顿 —— 快且稳(推荐对比用)
        """
        S = np.asarray(S, dtype=float)
        if cache is None:
            cache = self._cache(S)
        self.phi_mode_info = {"mode": mode, "n": int(np.prod(S.shape[:-1])),
                              "n_cubic_fallback": 0, "n_newton_fallback": 0, "n_bad": 0}
        if mode == "newton":
            return self.phi1se_newton(S, cache=cache)
        if mode == "cubic":
            return self.phi1se_newton(S, phi_start=self.phi1se_cubic(S, cache), cache=cache)
        if mode == "cubic_pure":
            phi = np.asarray(self.phi1se_cubic(S, cache), dtype=complex)
        elif mode in ("quartic_pure", "closed"):
            phi, ok = self.phi1se_quartic(S, cache=cache)
            if mode == "closed" and not np.all(ok):
                # 四次解不可用的点(退化/伪根)直接交给牛顿, **不能**用三次解兜底:
                # 三次解即使数值有限也可能是错的, 而边缘相位一旦错, (71)(72) 的
                # 过渡分支就会把 P_d 放大成垃圾(实测纯三次解会让方向图 +250 dB).
                self.phi_mode_info["n_newton_fallback"] = int(np.sum(~ok))
                Sf = S.reshape(-1, 3)
                mask = (~ok).ravel()
                flat = np.asarray(phi, dtype=complex).ravel().copy()
                flat[mask] = self._newton_subset(Sf, mask, cache)
                phi = flat.reshape(np.shape(phi))
                ok = np.ones_like(ok)
        else:
            raise ValueError(f"phi1se mode 只能是 newton/cubic/cubic_pure/quartic_pure/closed, "
                             f"收到 {mode!r}")
        bad = (~np.isfinite(phi) | (np.abs(np.imag(phi)) > 3.0)
               | (np.abs(np.real(phi)) > np.pi))
        if np.any(bad):
            self.phi_mode_info["n_bad"] = int(np.sum(bad))
            if mode == "closed":                       # 最后兜底: 牛顿
                self.phi_mode_info["n_newton_fallback"] = int(np.sum(bad))
                phi = np.where(bad, self.phi1se_newton(S), phi)
            else:                                      # 纯闭式: 退回 phi=0
                phi = np.where(bad, 0.0, phi)
        return phi

    def _newton_subset(self, S_flat, mask, cache):
        """只对 mask 选中的方向跑牛顿（其它方向的结果不动，省时间）。

        cache 只对应整体方向集，所以这里按子集重算一次 cache。
        """
        sub = self.phi1se_newton(S_flat[mask])
        return sub


    # ---------- 边缘点 Q 解冻 ----------
    def edge_q_correction(self, x, y):
        """把 (73) 边缘点的二次相位从 Q(0) 换成局部 Q(z_i):

            f = -R - z_i - 1/2 Q xi^2   ==>   df = -1/2 [Q(z_i)-Q(0)] xi^2

        注意: (19) 的振幅因子 sqrt(det Q(z_i)/det Q(0)) 本来就用的局部 z_i,
        所以只需要修相位这一项。鞍点处 xi=0, 修正自然为 0, 与 P_r 保持自洽。
        """
        lm, bm = self.lm, self.beam
        z = lm.zloc(x, y)
        ri = np.stack([x, y, z], axis=-1) @ bm.C.T
        xi, yi, zi = ri[..., 0], ri[..., 1], ri[..., 2]
        dQ = bm.Q(zi) - self.Q0
        return -0.5 * dQ * (xi * xi + yi * yi)

    # ---------- 主入口 ----------
    def Ps(self, S, phi1se_mode="newton"):
        """返回 P^s = r*E^s*exp(+jkr) 及各中间量"""
        lm = self.lm
        S = np.asarray(S, dtype=float)
        # 与 φ 无关的 9 个量只算一次, 供 phi1se 求解/_edge_terms/gpp_rho 共用 (A.20.1)
        cache = self._cache(S)
        a0, a1, b0, b1, cc, xs, ys, fs, den = cache

        # ---- (68) 反射场 ----
        Fs = self.fbar(xs, ys, S)
        w1 = spim(-1j * self.k * a1)
        w2 = spim(-1j * self.k * den / a1)
        Pr = (-np.pi / (w1 * w2))[..., None] * Fs * np.exp(1j * self.k * fs)[..., None]

        # ---- 附录 B: 边缘量 (B1)(B3)(B5) ----
        phi = self.phi1se(S, phi1se_mode, cache=cache)
        x_e, y_e, fx_e, fy_e, D_e, _ = self._edge_terms(S, phi, cache)
        Xr, Yr, Xrp, Yrp = lm.edge_map(phi)
        gpd = fx_e * Xr + fy_e * Yr                                                   # (B3)
        xd = lm.d * Xrp                # dx/dphi1  at rho1 = d
        yd = lm.d * Yrp
        gpp_phi = (-2.0 * a0 * xd ** 2 - 4.0 * cc * xd * yd - 2.0 * a1 * yd ** 2
                   - fx_e * lm.d * Xr - fy_e * lm.d * Yr)                              # 等价于 (B5)
        f_se = (fs - a0 * (x_e - xs) ** 2 - a1 * (y_e - ys) ** 2
                - 2.0 * cc * (x_e - xs) * (y_e - ys))                                  # (B1)
        if self.edge_q == "local":
            f_se = f_se + self.edge_q_correction(x_e, y_e)

        Fe = self.fbar(x_e, y_e, S)
        # (73): 系数 ABd -> d*|det M| (通用雅可比 |J(d)| = d*|A2*B1 - A1*B2|, 见 (53))
        with np.errstate(all="ignore"):
            coef_d = lm.d * lm.jac / (1j * self.k * gpd)
            coef_d = np.where(np.isfinite(coef_d), coef_d, 0.0)
            tail_d = (1j * SQRT2PI * np.exp(1j * self.k * f_se) / spim(1j * self.k * gpp_phi))
            tail_d = np.where(np.isfinite(tail_d), tail_d, 0.0)
        Pd = Fe * coef_d[..., None] * tail_d[..., None]                              # (73)

        # ---- (69)(71)(72): S_a 及其符号 ----
        # 用 (45) 的反变换取驻定点的 (rho1s, 方向):  (u,v) = M^-1 (x_s-x_o, y_s-y_o)
        dxs = xs - lm.xo
        dys = ys - lm.yo
        u = lm.Minv[0, 0] * dxs + lm.Minv[0, 1] * dys
        v = lm.Minv[1, 0] * dxs + lm.Minv[1, 1] * dys
        rho_s = np.sqrt(u * u + v * v)
        rho_s = np.where(np.real(rho_s) < 0.0, -rho_s, rho_s)
        ar = dxs / rho_s                 # = A cos(psi1s) 的通用形式
        br = dys / rho_s                 # = B sin(psi1s) 的通用形式
        gpp_rho = -2.0 * (a0 * ar ** 2 + a1 * br ** 2 + 2.0 * cc * ar * br)
        Sa = np.sqrt(1j * self.k * (fs - f_se))                                            # (69)
        # (71)(72): 用 Re[sqrt(...)] > 0 的分支(不是 spim 的 Im>=0)
        target = np.real(srep(-1j * self.k / 2.0 * gpp_rho) * (lm.d - rho_s))
        Sa = np.where(np.real(Sa) * target < 0.0, -Sa, Sa)

        # ---- (65) 过渡函数 ----
        # scipy 的复 erf 在 Re(z^2) < -709 时溢出成 ±inf(即 |Im S_a| 超过约 26.7),
        # 于是 0.5*(1+erf) = nan, 再乘上下溢到 ~1e-313 的 |P_r| 就得到 nan
        # （这就是 "invalid value encountered in multiply" 的来源）。
        # 那些方向距该束的镜面方向已经很远(实测 25°~90°, 取决于束宽), 物理上
        # 只有渐近极限有意义: 1/2[1+erf(S_a)] -> (Re S_a>0 ? 1 : 0), T -> (Re S_a>0 ? 1 : 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            erf_Sa = erf(Sa)
            fall = 0.5 * (1.0 + erf_Sa)
            T_raw = 1.0 - 0.5 * (1.0 - erf_Sa) + np.exp(-Sa ** 2) / (2.0 * SQRTPI * Sa)
            pos = np.real(Sa) > 0.0
            n_clip = int(np.sum(~np.isfinite(fall)) + np.sum(~np.isfinite(T_raw)))
            fall = np.where(np.isfinite(fall), fall, np.where(pos, 1.0, 0.0))
            T = np.where(np.isfinite(T_raw), T_raw, np.where(pos, 1.0, 0.0))
            bad_pr = ~np.isfinite(Pr)
            bad_pd = ~np.isfinite(Pd)
            if np.any(bad_pr):
                Pr = np.where(bad_pr, 0.0, Pr)
            if np.any(bad_pd):
                Pd = np.where(bad_pd, 0.0, Pd)

        Ps = T[..., None] * Pr + Pd                                                   # (74) 字面式
        # 数值稳定形式: T*P_r 与 P_d 在 S_a->0 时各自发散并相消, 留到
        # 领先阶正好是 P_r*(1/2)[1+erf(S_a)] (推导见 REPORT.md §3).
        Ps_stable = Pr * fall[..., None]
        # 自动混合式: |S_a| 远离 0 时用字面式(实测更准, 见 REPORT.md §A.11),
        # |S_a| -> 0 的阴影边界处切换到稳定式(字面式在此发散), 中间平滑过渡.
        # 两项在过渡区都是有限值(字面式只在 |S_a|≲1e-8 才炸), 所以加权混合安全.
        # 字面式的奇点兜底: S_a -> 0 时用稳定式顶替(两项发散、相消靠舍入)
        near = np.abs(Sa) < self.form_eps
        n_near = int(np.sum(near))
        if n_near:
            Ps = np.where(near[..., None], Ps_stable, Ps)
        A_LO, A_HI = self.form_lo, self.form_hi
        wgt = np.clip((np.abs(Sa) - A_LO) / max(A_HI - A_LO, 1e-12), 0.0, 1.0)
        wgt = wgt * wgt * (3.0 - 2.0 * wgt)                     # smoothstep
        wgt = np.where(np.isfinite(Ps).all(axis=-1), wgt, 0.0)  # 字面式非有限处直接退回稳定式
        Ps_auto = (wgt[..., None] * Ps + (1.0 - wgt[..., None]) * Ps_stable)

        # ---- 纯解析路径 (REPORT A.27) --------------------------------------
        #   P_analytic = P_r · [ T_straight − S_ratio ]
        #     T_straight = ½[1+erf(Z)],  Z = √c_eff·d_r   （直边, 只用实几何距离）
        #     S_ratio    = erf 展开 + 复高斯精确取矩      （月牙/曲边, 初等闭式项）
        # 三块都是初等闭式：反射本体 P_r 是二次型 SPP，直边是 erf 解析式，
        # 月牙是一串初等项。**整条链不含任何复根求解**，也不需要窗/上限——
        # 因为 Z 只用实几何 d_r，Re Z 与 d_r 同号，T_straight 单调 (0,1) 有界。
        # 关掉时 Ps_analytic ≡ Ps_stable（逐位相同），方便一次跑出 A/B。
        if self.edge_crescent:
            cr = self.crescent_ratio(S, cache=cache)
            S_ratio_raw = cr["S_ratio"]
            T_analytic = cr["T_straight"] - S_ratio_raw
            Ps_analytic = Pr * T_analytic[..., None]
            bad = ~np.isfinite(Ps_analytic).all(axis=-1)
            if np.any(bad):
                Ps_analytic = np.where(bad[..., None], Ps_stable, Ps_analytic)

            # ---- 嫁接式(仅对照，**不是**推荐路径)：把 S_ratio 加到现有的 fall 上 ----
            # 现有 fall=½[1+erf(S_a)] 里的 S_a 来自 (B7) 的复根，|Z| 大时它会随 Im S_a
            # 指数增长(不再是截断因子)，所以这里只能加窗+上限才安全。A.27 记录了它的表现。
            zt = np.abs(cr["Z"])
            win = np.clip(2.0 - zt, 0.0, 1.0)
            win = win * win * (3.0 - 2.0 * win)          # smoothstep: |Z|<=1 全用, >=2 关闭
            S_ratio_w = S_ratio_raw * win
            lim = 0.5 * np.maximum(np.abs(fall), 1e-12)
            mag = np.abs(S_ratio_w)
            S_ratio_w = S_ratio_w * np.minimum(1.0, lim / np.maximum(mag, 1e-30))
            T_crescent = fall - S_ratio_w
            Ps_crescent = Pr * T_crescent[..., None]
            bad = ~np.isfinite(Ps_crescent).all(axis=-1)
            if np.any(bad):
                Ps_crescent = np.where(bad[..., None], Ps_stable, Ps_crescent)
            S_ratio = S_ratio_raw
            cr_d, cr_k = cr["d_r"], cr["kappa"]
            cr_ok = cr["ok"]
        else:
            S_ratio = np.zeros_like(fall, dtype=complex)
            T_crescent = fall
            Ps_crescent = Ps_stable
            T_analytic = fall
            Ps_analytic = Ps_stable
            cr_d = np.zeros_like(fall, dtype=float)
            cr_k = np.zeros_like(fall, dtype=float)
            cr_ok = np.zeros(np.shape(fall), dtype=bool)

        return {
            "Ps": Ps, "Ps_stable": Ps_stable, "Ps_auto": Ps_auto,
            "Ps_crescent": Ps_crescent, "Ps_analytic": Ps_analytic,
            "T_analytic": T_analytic, "S_ratio": S_ratio,
            "T_crescent": T_crescent, "d_r": cr_d, "kappa_rim": cr_k,
            "crescent_ok": int(np.count_nonzero(cr_ok)),
            "Pr": Pr, "Pd": Pd, "T": T, "Sa": Sa, "w_auto": wgt,
            "n_clip": n_clip, "n_near": n_near,
            "im_Sa_max": float(np.max(np.abs(np.imag(Sa)))),
            "xs": xs, "ys": ys, "fs": fs, "den": den,
            "a0": a0, "a1": a1, "b0": b0, "b1": b1, "c": cc,
            "phi1se": phi, "rho1s": rho_s, "gpd": gpd, "gpp_phi": gpp_phi,
            "gpp_rho": gpp_rho, "f_se": f_se, "x_e": x_e, "y_e": y_e,
            "Fbar_s": Fs, "Fbar_e": Fe,
        }


# --------------------------------------------------------------------------- #
# 纯解析路径 (REPORT A.27)
# --------------------------------------------------------------------------- #
class ChouPathakAnalytic:
    """chou1997 反射的**纯解析路径**：三块全是初等闭式，不含任何复根求解。

        反射本体   P_r = (-pi/(w1 w2)) F_s e^{jk f_s}          ← 二次型 SPP，初等
        直边截断   T_st = ½[1+erf(Z)],
                   Z = sqrt(c_eff) d_r - jk b_eff/(2 sqrt(c_eff))   ← 含 erf 的解析式
        月牙修正   S_ratio = e^{-Z²} sum_n λ^n H_{n-1}(Z) M_{2n}/n!  ← 初等项
        T = T_st - S_ratio,  P = P_r · T

    与 `ChouPathakSingle` 相比，边缘点**不靠 (B7)/(B13) 的复根求 φ₁ₛ**：
    - (a_ξξ, a_ξη, a_ηη)、(b_ξ, b_η) 是 f 在局部系里的二次/线性系数，
      旋到 (e_r, e_φ) 后做 Schur 补：  c_eff = jk(a_ξξ - a_ξη²/a_ηη)，
                                      b_eff = b_ξ - a_ξη b_η/a_ηη；
    - d_r, κ 是**每束一次**的实几何量（原点 = 束的反射点 = 光斑中心），
      不随观察方向变；方向依赖全部由 b = S - S_spec 承担（S_spec = 镜面方向 = C 的第 3 行）。
    - 实测(REPORT A.27)：这个闭式 Z 与四次解出的 S_a 在 0~30° 上**量级逐位相等**
      （1°: 0.550/0.550 … 30°: 1.16e14/1.16e14），只差一个整体共轭（相位约定）。

    因此 T 仍会随 |Im Z| 指数增长——这不是病态，而是它抵消 P_r 的束斑衰减、
    承载宽角绕射能量的方式。

    参数
    ----
    conjugate : 取 Z 的共轭（相位约定二选一，用端到端 PO 对拍定，见 A.27）
    order     : 月牙级数阶数（实测 1 与 2 逐位相同）
    """

    def __init__(self, lm, beam, k=None, order=1, conjugate=False):
        self.spp = ChouPathakSingle(lm, beam, k=k)
        self.lm = self.spp.lm
        self.beam = self.spp.beam
        self.k = self.spp.k
        self.order = int(order)
        self.conjugate = bool(conjugate)
        self._setup()

    # ---- 每束一次的实几何 + 解析系数 ----
    def _setup(self):
        spp, lm = self.spp, self.lm
        self.Sspec = np.asarray(spp.c[2], dtype=float)        # (c31,c32,c33) 镜面方向(局部)
        self.ok = bool(np.isfinite(lm.d) and lm.d > 0.0
                       and not bool(getattr(lm, "d_imag", False)))
        if not self.ok:
            self.d_r, self.kappa = 0.0, 0.0
            self.e_r = np.array([1.0, 0.0])
            self.e_p = np.array([0.0, 1.0])
            self.a_frozen = (0j, 0j, 0j)
            self.c_eff0 = 1.0 + 0.0j
            self.a_pp0 = 1.0 + 0.0j
            self.lam0 = 0.0j
            return
        a0, a1, b0, b1, cc = spp.coeffs(self.Sspec)           # 镜面方向: b0=b1=0
        self.a_frozen = (complex(a0), complex(a1), complex(cc))
        gm = spp.rim_nearest(0.0, 0.0)                        # 原点 = 束的反射点
        self.d_r = float(np.real(gm["d_r"]))
        self.kappa = float(np.real(gm["kappa"]))
        self.e_r = np.asarray(gm["er"], dtype=float).reshape(-1)[:2]
        self.e_p = np.asarray(gm["ep"], dtype=float).reshape(-1)[:2]
        a_rr, a_rp, a_pp = self._rotate(self.a_frozen)
        self.a_pp0 = complex(a_pp)
        self.c_eff0 = 1j * self.k * (a_rr - a_rp ** 2 / a_pp)

    def _rotate(self, a):
        """二次型 (a0,a1,cc) 旋到 (e_r,e_φ)"""
        a0, a1, cc = a
        ex, ey = self.e_r
        px_, py_ = self.e_p
        a_rr = a0 * ex * ex + 2.0 * cc * ex * ey + a1 * ey * ey
        a_rp = a0 * ex * px_ + cc * (ex * py_ + ey * px_) + a1 * ey * py_
        a_pp = a0 * px_ * px_ + 2.0 * cc * px_ * py_ + a1 * py_ * py_
        return a_rr, a_rp, a_pp

    # ---- 每方向: 反射本体 + 直边 + 月牙 ----
    def Ps(self, S):
        S = np.atleast_2d(np.asarray(S, dtype=float))
        spp, k = self.spp, self.k
        cache = spp._cache(S)
        a0, a1, b0, b1, cc, xs, ys, fs, den = cache

        # (1) 反射本体: 鞍点处二次型的 SPP —— 初等闭式
        with np.errstate(all="ignore"):
            Fs = spp.fbar(xs, ys, S)
            w1 = spim(-1j * k * a1)
            w2 = spim(-1j * k * (den / a1))
            Pr = (-np.pi / (w1 * w2))[..., None] * Fs * np.exp(1j * k * fs)[..., None]

        if not self.ok:
            T = np.ones(np.shape(np.asarray(xs)))
            Z = np.zeros(np.shape(T), dtype=complex)
            S_ratio = np.zeros(np.shape(T), dtype=complex)
        else:
            # (2) 直边: Z = sqrt(c_eff) d_r - jk b_eff / (2 sqrt(c_eff))
            a_rr, a_rp, a_pp = self._rotate((a0, a1, cc))
            c_eff = 1j * k * (a_rr - a_rp ** 2 / a_pp)
            sq = srep(c_eff)                                  # Re sqrt >= 0
            b_r = b0 * self.e_r[0] + b1 * self.e_r[1]
            b_p = b0 * self.e_p[0] + b1 * self.e_p[1]
            b_eff = b_r - a_rp * b_p / a_pp
            with np.errstate(all="ignore"):
                Z = sq * self.d_r - 1j * k * b_eff / (2.0 * sq)
            if self.conjugate:
                Z = np.conj(Z)
            T_st = 0.5 * (1.0 + erf(Z))
            # (3) 月牙: erf 展开 + 复高斯精确取矩（初等项）
            c_p = 1j * k * a_pp
            lam = sq * self.kappa / 2.0
            s2 = 1.0 / (2.0 * c_p)
            acc = lam * s2
            if self.order >= 2:
                acc = acc + lam ** 2 * (2.0 * Z) * (3.0 * s2 ** 2) / 2.0
            if self.order >= 3:
                acc = acc + lam ** 3 * (4.0 * Z ** 2 - 2.0) * (15.0 * s2 ** 3) / 6.0
            with np.errstate(all="ignore"):
                S_ratio = np.exp(-Z * Z) * acc / SQRTPI
            S_ratio = np.where(np.isfinite(S_ratio), S_ratio, 0.0)
            T = T_st - S_ratio
            T = np.where(np.isfinite(T), T, T_st)

        P = Pr * T[..., None]
        bad = ~np.isfinite(P).all(axis=-1)
        if np.any(bad):
            P = np.where(bad[..., None], 0.0, P)
        return {"Ps": P, "Ps_stable": P, "Ps_auto": P, "Ps_crescent": P,
                "Ps_analytic": P, "Pr": Pr, "T": T, "T_analytic": T,
                "T_crescent": T, "S_ratio": S_ratio, "Sa": Z, "Z": Z,
                "d_r": np.full(np.shape(np.real(np.asarray(xs))), self.d_r),
                "kappa_rim": np.full(np.shape(np.real(np.asarray(xs))), self.kappa),
                "n_clip": 0, "n_near": 0, "n_transition": 0,
                "phi1se": None, "P_r": Pr}


# --------------------------------------------------------------------------- #
# 数值 PO 金标准
# --------------------------------------------------------------------------- #
def po_far_field(parab, lm, beam, S_list, n_rr=260, n_ph=1024, model="reflector", k=None):
    """数值 PO 远场 (Omega = r*E^s*exp(+jkr)):

        Omega = (jkZ0/4pi) * int [S x (S x J_eq)] exp(jk S.r') dS'        (9)(11)

    model='reflector': 实际抛物面圆盘 (0<=rho_r<=a)
    model='local'    : 论文的局部模型 (局部抛物面 (2) + 投影椭圆边缘),
                       用来把"局部近似误差"与"渐近求值误差"分开
    k=None 时用模块默认 K(长度为波长单位); GBMA 里请显式传入真实波数。
    """
    kk = K if k is None else float(k)
    R_l = np.stack([lm.xhat, lm.yhat, lm.zhat])       # 行 = 局部基(全局分量)
    if model == "reflector":
        nodes, wts = np.polynomial.legendre.leggauss(n_rr)
        rr = 0.5 * parab.a * (nodes + 1.0)
        wr = 0.5 * parab.a * wts
        pp = 2.0 * np.pi * np.arange(n_ph) / n_ph
        R_, P_ = np.meshgrid(rr, pp, indexing="ij")
        X = R_ * np.cos(P_)
        Y = R_ * np.sin(P_)
        Z = (X * X + Y * Y) / (4.0 * parab.F)
        Pts = np.stack([X, Y, Z], axis=-1)
        nrm = parab.normal(X, Y)
        dS = np.sqrt(1.0 + (R_ / (2.0 * parab.F)) ** 2) * R_ * (wr[:, None] * (2.0 * np.pi / n_ph))
        Pl = lm.to_local(Pts)
        # 束场在**局部基**里, 必须先旋到全局基再与全局法向叉乘
        H = beam.H(Pl) @ R_l
        J = 2.0 * np.cross(nrm, H)                                    # (11)
        Wt = (J * dS[..., None]).reshape(-1, 3)
        Pf = Pts.reshape(-1, 3)
    else:
        nodes, wts = np.polynomial.legendre.leggauss(n_rr)
        rr = 0.5 * lm.d * (nodes + 1.0)
        wr = 0.5 * lm.d * wts
        pp = -np.pi + 2.0 * np.pi * np.arange(n_ph) / n_ph
        R_, P_ = np.meshgrid(rr, pp, indexing="ij")
        X, Y = lm.xy(R_, P_)
        Z = lm.zloc(X, Y)
        nv = lm.surf_nvec(X, Y)
        nrm = nv / np.linalg.norm(nv, axis=-1, keepdims=True)
        # dS = sqrt(1+(x/R1)^2+(y/R2)^2) dx dy,  dx dy = |J| d rho1 d phi1
        # |J| = rho1 * |det M| = rho1 * lm.jac  (53) 的通用形式
        jac = lm.jac * R_
        dS = np.linalg.norm(nv, axis=-1) * jac * (wr[:, None] * (2.0 * np.pi / n_ph))
        Pl = np.stack([X, Y, Z], axis=-1)
        H = beam.H(Pl) @ R_l
        J = 2.0 * np.cross(nrm, H)
        Wt = (J * dS[..., None]).reshape(-1, 3)
        Pf = lm.to_global(Pl).reshape(-1, 3)

    # 观察方向从"局部系"转到"反射面坐标系": S_global = S . [x_hat; y_hat; z_hat]
    Sg = np.asarray(S_list, dtype=float) @ R_l              # (M,3) 反射面坐标系
    Pft = Pf.T                                              # (3,N)
    out = np.empty((len(Sg), 3), dtype=complex)
    step = 8                                                # 分块控制内存
    for i0 in range(0, len(Sg), step):
        ph = np.exp(1j * kk * (Sg[i0:i0 + step] @ Pft))    # (m,N)
        tmp = ph @ Wt                                       # (m,3) = int J e^{jkS.r'} dS'
        Sh = Sg[i0:i0 + step]
        # PO 远场必须投影: E ~ S x (S x int J ...)  (之前漏了这一步, 仅在 S.J≈0 时无害)
        out[i0:i0 + step] = np.cross(Sh, np.cross(Sh, tmp))
    return out * (1j * kk * Z0 / (4.0 * np.pi))


# --------------------------------------------------------------------------- #
# 自检工具
# --------------------------------------------------------------------------- #
def _cstep(f, x, h=1e-5):
    """中心差分一阶导数.

    注意: 因为 Q 为复数, g/f 都是**复值**解析函数, 复步长技巧
    (Im f(x+ih)/h) 不适用, 这里用中心差分 (误差 ~h^2).
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def g_quad(cp, S, rho, phi):
    """(52)(33): g(rho1,phi1) = f(x',y') 的二次型版本"""
    lm = cp.lm
    a0, a1, _, _, cc = cp.coeffs(S)
    xs, ys, fs, _ = cp.stationary(S)
    x, y = lm.xy(rho, phi)
    return fs - a0 * (x - xs) ** 2 - a1 * (y - ys) ** 2 - 2.0 * cc * (x - xs) * (y - ys)


def check_stationary_and_derivs(cp, S_one, verbose=True):
    """(34)(35) 驻定相位点、(B3) g'_rho(d)、(B5) g''_phi(phi1se) 的对拍"""
    lm = cp.lm
    a0, a1, b0, b1, cc = cp.coeffs(S_one)
    xs, ys, fs, den = cp.stationary(S_one)
    f = lambda X, Y: fs - a0 * (X - xs) ** 2 - a1 * (Y - ys) ** 2 - 2.0 * cc * (X - xs) * (Y - ys)
    out = {}
    out["df_dx_at_xs"] = _cstep(lambda t: f(t, ys), xs)
    out["df_dy_at_ys"] = _cstep(lambda t: f(xs, t), ys)

    d = cp.Ps(np.asarray(S_one, float))
    phi = complex(np.asarray(d["phi1se"]).reshape(-1)[0])
    out["phi1se_deg"] = float(np.degrees(np.real(phi)))
    out["phi1se_imag"] = float(np.imag(phi))

    gpd = complex(np.asarray(d["gpd"]).reshape(-1)[0])
    gpd_fd = _cstep(lambda t: g_quad(cp, S_one, t, phi), lm.d)
    out["gpd_rel_err"] = float(abs(gpd - gpd_fd) / max(abs(gpd), 1e-300))

    gpp = complex(np.asarray(d["gpp_phi"]).reshape(-1)[0])
    dd = 1e-4
    f2 = lambda t: g_quad(cp, S_one, lm.d, t)
    gpp_fd = (_cstep(f2, complex(np.real(phi) + dd, np.imag(phi)))
              - _cstep(f2, complex(np.real(phi) - dd, np.imag(phi)))) / (2 * dd)
    out["gpp_phi_rel_err"] = float(abs(gpp - gpp_fd) / max(abs(gpp), 1e-300))

    psi = phi + lm.phi0
    gpp_B5 = (-lm.d * gpd
              - 2.0 * lm.d ** 2 * (a0 * lm.A ** 2 * np.sin(psi) ** 2
                                   - 2.0 * cc * lm.A * lm.B * np.sin(psi) * np.cos(psi)
                                   + a1 * lm.B ** 2 * np.cos(psi) ** 2))
    out["B5_rel_err"] = float(abs(gpp - gpp_B5) / max(abs(gpp), 1e-300))

    phi_cubic = np.asarray(cp.phi1se_cubic(np.asarray(S_one, float))).reshape(-1)[0]
    out["cubic_minus_newton"] = complex(phi_cubic - phi)
    x, y, fx, fy, D, _ = cp._edge_terms(np.asarray(S_one, float), np.array([phi_cubic]))
    out["B7_residual_cubic"] = float(abs(D[0]))
    xs_c = complex(np.asarray(xs).reshape(-1)[0])
    ys_c = complex(np.asarray(ys).reshape(-1)[0])
    rho_c = complex(np.asarray(d["rho1s"]).reshape(-1)[0])
    out["rho1s_re"] = float(np.real(rho_c))
    out["rho1s_minus_d"] = float(np.real(rho_c) - lm.d)

    if verbose:
        print("  [stationary / derivative checks]")
        print(f"    df/dx'(x_s)={out['df_dx_at_xs']:.2e}   df/dy'(y_s)={out['df_dy_at_ys']:.2e}")
        print(f"    x_s={xs_c:.4f}   y_s={ys_c:.4f}")
        print(f"    rho1s={rho_c:.4f}   d={lm.d:.4f}   Re(rho1s)-d={out['rho1s_minus_d']:+.3e}")
        print(f"    phi1se={out['phi1se_deg']:8.4f} deg  Im={out['phi1se_imag']:+.3e}   "
              f"|附录B cubic - Newton|={abs(out['cubic_minus_newton']):.3e}  "
              f"(B7 残差 {out['B7_residual_cubic']:.1e})")
        print(f"    (B3) 相对误差={out['gpd_rel_err']:.2e}   "
              f"(B5) 相对误差={out['gpp_phi_rel_err']:.2e} (显式式 {out['B5_rel_err']:.2e})")
    return out


def check_SPP_2d(cp, S_one, n=None, box=None, verbose=True):
    """V1: 把 (68) 与直接二维数值积分 int int exp(jkf) 对拍 (无边缘)

    积分区/节点数按衰减尺度自动选取(否则换成米的单位后原 box=60 会完全欠采样)。
    """
    a0, a1, b0, b1, cc = cp.coeffs(S_one)
    if box is None or n is None:
        decay = 1.0 / np.sqrt(max(cp.k * abs(np.imag(a0)), 1e-12))
        if box is None:
            box = 6.0 * decay
        if n is None:
            n = 900
    xs, ys, fs, den = cp.stationary(S_one)
    Fs = np.asarray(cp.fbar(xs, ys, S_one)).reshape(-1)
    nodes, wts = np.polynomial.legendre.leggauss(n)
    X = box * nodes
    W = box * wts
    XX, YY = np.meshgrid(X, X, indexing="ij")
    WW = np.outer(W, W)
    dX = XX - xs
    dY = YY - ys
    ph = np.exp(1j * cp.k * (fs - a0 * dX ** 2 - a1 * dY ** 2 - 2.0 * cc * dX * dY))
    val = np.sum(WW * ph)
    Pr = np.asarray(cp.Ps(np.asarray(S_one, float))["Pr"]).reshape(-1)
    k0 = int(np.argmax(np.abs(Fs)))          # F 只有主极化分量非零, 取该分量
    closed = Pr[k0] / Fs[k0]
    ratio = val / closed
    if verbose:
        print("  [V1: (68) vs 2D numerical SPP]")
        print(f"    数值 int int e^(jkf) = {val:.8e}")
        print(f"    闭式 -pi/(sqrt(-jka1)*sqrt(-jk(a0a1-c^2)/a1)) = {closed:.8e}   比值 = {ratio:.10f}")
    return {"spp_ratio": ratio, "spp_rel_err": float(abs(ratio - 1.0))}


def check_phase_model(cp, span=None, n=61, verbose=True):
    """量化约定 C1（把 Q 冻结成 Q(0)）带来的模型误差。

    (20) 是精确的 H^i, 但 (27) 要求 f 是二次型, 所以 (28)-(32) 里必须用常数
    Q = Q(0); 两者之差是一个纯粹的附加因子
        Δ = -jk/2 [Q(z_i) - Q(0)] ξ_i²
    在光斑上采样统计 |Δ| 与 |e^Δ - 1| 即可看到 C1 的量级。
    """
    lm, bm = cp.lm, cp.beam
    if span is None:
        span = 2.0 * bm.w(0.0)
    gx = np.linspace(-span, span, n)
    X, Y = np.meshgrid(gx, gx)
    keep = (X * X + Y * Y) <= span * span
    x = X[keep]
    y = Y[keep]
    z = lm.zloc(x, y)
    ri = np.stack([x, y, z], axis=-1) @ bm.C.T
    xi, yi, zi = ri[:, 0], ri[:, 1], ri[:, 2]
    Qz = bm.Q(zi)
    dQ = Qz - cp.Q0
    Delta = -1j * cp.k * 0.5 * dQ * (xi * xi + yi * yi)
    err = np.abs(np.exp(Delta) - 1.0)
    w = np.exp(-2.0 * (xi * xi + yi * yi) / (bm.w(0.0) ** 2))     # 按光强加权
    if verbose:
        print("  [C1: Q 冻结成 Q(0) 的模型误差]")
        print(f"    |Δ| : median={np.median(np.abs(Delta)):.3e} rad, "
              f"95%={np.percentile(np.abs(Delta), 95):.3e} rad")
        print(f"    |e^Δ-1| (按光强加权 RMS) = {np.sqrt(np.sum(w * err ** 2) / np.sum(w)):.3e}")
    return {"phase_model_rms": float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))}


# --------------------------------------------------------------------------- #
# 算例
# --------------------------------------------------------------------------- #
def build_case(F=36.0, a=20.0, r0=20.0, b=12.53, pol_ref=None, rot_deg=0.0, verbose=True):
    """按论文 Fig.5-7 的量级取参数: F=36, D=40 (a=20), b=12.53 (w0 约 2).
    束腰放在焦点, 束轴由焦点指向 O; O 距反射面轴 r0.

    rot_deg != 0 时把束轴绕局部 ẑ 旋转, 使入射面偏离子午面 ->
    (7) 的 c12,c21 非零, 用于检验 (28)-(32) 的交叉项与 (B7)-(B23).
    """
    parab = Paraboloid(F, a)
    lm = LocalModel(parab, r0)
    focus = np.array([0.0, 0.0, F])
    axis = lm.O - focus
    L = float(np.linalg.norm(axis))
    axis = axis / L
    zi_local = np.array([axis @ lm.xhat, axis @ lm.yhat, axis @ lm.zhat])
    if rot_deg:
        t = np.deg2rad(rot_deg)
        Rz = np.array([[np.cos(t), -np.sin(t), 0.0],
                       [np.sin(t), np.cos(t), 0.0],
                       [0.0, 0.0, 1.0]])
        zi_local = Rz @ zi_local
    beam = GaussianBeam(b, L, zi_local, pol_ref=(lm.xhat if pol_ref is None else pol_ref))
    cp = ChouPathakSingle(lm, beam)
    gamma = float(np.degrees(np.arccos(np.clip(-zi_local[2], -1.0, 1.0))))
    if verbose:
        print(f"\n===== 算例: F={F:g}, a={a:g} (D={2*a:g}), r0={r0:g}, "
              f"b={b:g} (w0={beam.w0:.3f}), 入射面旋转 {rot_deg:g} deg =====")
        print(f"  入射角 gamma={gamma:.3f} deg, 束腰在焦点, L={L:.3f}, "
              f"w(O)={beam.w(0.0):.3f}, c33={zi_local[2]:.4f}, "
              f"c12={beam.C[0, 1]:+.4f}, c21={beam.C[1, 0]:+.4f}")
    lm.check(verbose=verbose)
    beam.check(verbose=verbose)
    return parab, lm, beam, cp, dict(gamma_deg=gamma, L=L)


def S_from_cut(theta_deg, phi_deg):
    th = np.deg2rad(np.asarray(theta_deg, float))
    ph = np.deg2rad(phi_deg)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)


def pow_dB(vec):
    return 10.0 * np.log10(np.sum(np.abs(vec) ** 2, axis=-1) + 1e-300)


def compare_cut(parab, lm, beam, cp, theta_deg, phi_deg, n_rr=260, n_ph=1024,
                model="reflector", label="", form="stable"):
    S = S_from_cut(theta_deg, phi_deg)
    d = cp.Ps(S)
    th = np.asarray(pow_dB(d["Ps_stable"] if form == "stable" else d["Ps"]))
    th_lit = np.asarray(pow_dB(d["Ps"]))
    po = po_far_field(parab, lm, beam, S, n_rr=n_rr, n_ph=n_ph, model=model)
    pth = np.asarray(pow_dB(po))
    return dict(S=S, theory=d, th=th, th_literal=th_lit, po=po, pth=pth,
                theta=np.asarray(theta_deg, float), phi=phi_deg, label=label)


def summarize(cuts, tag="", local=None):
    print(f"\n  --- 对拍结果 {tag} ---")
    print(f"  {'phi':>6} {'峰(dB,理论)':>13} {'峰(dB,PO)':>12} {'峰值差':>8} "
          f"{'RMS(主瓣5deg)':>14} {'RMS(全)':>10} {'RMS(vs局部模型PO)':>18}")
    rows = []
    for i, c in enumerate(cuts):
        dlt = c["th"] - c["pth"]
        mask = c["pth"] > (c["pth"].max() - 60.0)
        i0 = int(np.argmax(c["pth"]))
        core = mask & (np.abs(c["theta"] - c["theta"][i0]) <= 5.0)
        rms_all = float(np.sqrt(np.mean(dlt[mask] ** 2))) if mask.any() else float("nan")
        rms_core = float(np.sqrt(np.mean(dlt[core] ** 2))) if core.any() else float("nan")
        rms_loc = float("nan")
        if local is not None:
            ploc = local[i]
            m2 = ploc > ploc.max() - 60.0
            rms_loc = float(np.sqrt(np.mean((c["th"] - ploc)[m2] ** 2)))
        print(f"  {c['phi']:>5g}d {c['th'].max():>13.2f} {c['pth'].max():>12.2f} "
              f"{c['th'].max()-c['pth'].max():>8.2f} {rms_core:>14.2f} {rms_all:>10.2f} "
              f"{rms_loc:>18.2f}")
        rows.append(dict(phi=c["phi"], peak_theory=float(c["th"].max()),
                         peak_po=float(c["pth"].max()),
                         rms_core=rms_core, rms_all=rms_all, rms_local=rms_loc))
    return rows


def extra_cases(theta=None, verbose=True):
    """扩展检验:
    C) 入射面绕 ẑ 旋转 40° -> (7) 的 c12/c21 非零, 检验交叉项与 (B7)-(B23)
    D) 束轴落在反射面外 (r0=28>a=20) -> 只有局部解析延拓 + 边缘绕射, 是最难的区
    """
    if theta is None:
        theta = np.linspace(-10.0, 50.0, 121)
    out = {}
    for name, kw, phis in (
        ("C 入射面旋转 40deg (r0=20)", dict(r0=20.0, rot_deg=40.0), (20.0, 40.0, 90.0)),
        ("D 束轴在面外 (r0=28)", dict(r0=28.0), (0.0, 90.0)),
    ):
        parab, lm, beam, cp, geo = build_case(verbose=verbose, **kw)
        for phi in phis:
            S = S_from_cut(theta, phi)
            d = cp.Ps(S)
            thd = np.asarray(pow_dB(d["Ps_stable"]))
            po = np.asarray(pow_dB(po_far_field(parab, lm, beam, S, n_rr=260, n_ph=1024)))
            m = po > po.max() - 60.0
            rms = float(np.sqrt(np.mean((thd - po)[m] ** 2)))
            print(f"    phi={phi:5.1f}  峰 {thd.max():7.2f} vs {po.max():7.2f} dB "
                  f"(差 {thd.max()-po.max():+5.2f})  RMS={rms:5.2f} dB")
            out[(name, phi)] = dict(rms=rms, peak=float(thd.max() - po.max()))
    return out


def main():
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [
        ("A: O in surface (r0=5)", dict(r0=5.0)),
        ("B: beam axis on rim (r0=20)", dict(r0=20.0)),
    ]
    theta = np.linspace(-10.0, 50.0, 121)
    fig, axes = plt.subplots(len(cases), 2, figsize=(12.5, 4.3 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])

    for i, (name, kw) in enumerate(cases):
        parab, lm, beam, cp, geo = build_case(verbose=True, **kw)
        S0 = S_from_cut(np.array([max(geo["gamma_deg"], 1.0)]), 0.0)[0]
        print("  [单方向自检]")
        check_stationary_and_derivs(cp, S0, verbose=True)
        check_SPP_2d(cp, S0, verbose=True)
        check_phase_model(cp, verbose=True)

        cuts = []
        for phi_deg in (0.0, 90.0):
            cuts.append(compare_cut(parab, lm, beam, cp, theta, phi_deg, label=name))
        # 与"局部模型"的数值 PO 比: 只差渐近求值, 不含抛物面/边缘的局部近似
        local = []
        for c in cuts:
            loc = po_far_field(parab, lm, beam, c["S"], n_rr=260, n_ph=1024, model="local")
            local.append(np.asarray(pow_dB(loc)))
        summarize(cuts, tag=name, local=local)

        for j, c in enumerate(cuts):
            ax = axes[i, j]
            pk = max(c["th"].max(), c["pth"].max())
            ax.plot(c["theta"], c["th"] - pk, "b-", lw=1.3, label="Chou-Pathak (74), stable form")
            ax.plot(c["theta"], c["pth"] - pk, "r--", lw=1.0, label="numerical PO")
            ax.plot(c["theta"], c["th_literal"] - pk, "m:", lw=0.8, label="literal T*P_r+P_d")
            pr = np.asarray(pow_dB(np.asarray(c["theory"]["Pr"])))
            ax.plot(c["theta"], pr - pk, "g:", lw=0.9, label="P_r only (68)")
            ax.set_title(f"{name}   phi={c['phi']:g} deg")
            ax.set_xlabel("theta (deg) from local z")
            ax.set_ylabel("r*|E|^2 (dB)")
            ax.set_ylim(-60, 3)
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=7)

    fig.tight_layout()
    out = "gb_single_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图片已保存: {os.path.abspath(out)}")

    print("\n===== 扩展检验 (入射面旋转 / 束轴在面外) =====")
    extra_cases(theta=theta)


if __name__ == "__main__":
    main()
