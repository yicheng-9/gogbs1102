import numpy as np
import time
import contextlib
import io
from scipy.optimize import brentq
from dataclasses import dataclass

from feed_antenna import FeedAntenna
from mesh_generator import generate_parabolic_mesh, compute_incident_H
from po_integrator_constant import POIntegrator
from gbma import compute_far_field_gbma, po_reference, gbma_far_field, directions

# ========== 辅助函数 ==========
def compute_feed_total_power(feed):
    """计算馈源总辐射功率（用于方向性归一化）"""
    x = feed.k * feed.b
    P_total = (np.pi / (8.0 * x**3)) * (8.0*x**2 - 4.0*x + 1.0 - np.exp(-4.0*x))
    return P_total

def compute_theta_b_from_feed(feed, tol_deg=0.01, max_deg=90):
    """根据馈源远场幅度方向图计算半功率波束宽度 θ_b（度）"""
    target = 1.0 / np.sqrt(2.0)
    theta_test = np.linspace(0, max_deg, 500)
    amp_test = feed.far_field_amplitude(np.deg2rad(theta_test))
    idx = np.where(amp_test <= target)[0]
    if len(idx) == 0:
        return max_deg, np.deg2rad(max_deg)
    idx0 = idx[0]
    if idx0 == 0:
        return compute_theta_b_from_feed(feed, tol_deg, max_deg*2)
    a_deg = theta_test[idx0-1]
    b_deg = theta_test[idx0]
    def func(deg):
        return feed.far_field_amplitude(np.deg2rad(deg)) - target
    try:
        theta_b_deg = brentq(func, a_deg, b_deg, xtol=tol_deg/100)
    except ValueError:
        theta_b_deg = (a_deg + b_deg) / 2.0
    return theta_b_deg, np.deg2rad(theta_b_deg)

def po_mesh_sizes(cfg, mult=1, verbose=None):
    """数值积分网格的 (N_radial, N_azimuth) 与以波长为单位的步长。

    cfg.po_n_radial / po_n_azimuth > 0 时直接采用；否则用自动公式
        z  = 1.09·(πD/λ)·sin(theta0_max) + 10
        N_azimuth = z,  N_radial = z/2.4
    自动公式只按**扫描范围**定网格，不保证宽角精度，所以最后再用
    cfg.po_max_step_lam 封顶两个方向的步长（默认 0.5λ）。

    为什么必须封顶（实测，20 GHz/D=1，PO 在 30° 处的电平）：
        方位步长 2.38λ（88 边形近似圆边缘） → 30° 处 −40.8 dB  ← 伪散射瓣
        方位步长 1.05λ                        → −70.2 dB
        方位步长 0.52λ                        → −70.0 dB（收敛）
    2.38λ 会在 ~λ/(2.38λ) ≈ 24° 处造出 +30 dB 的假瓣，宽角对比会完全失真。
    """
    if cfg.po_n_radial > 0:
        nr = cfg.po_n_radial
    else:
        z = (1.09 * (np.pi * cfg.D / cfg.lam)
             * np.sin(np.deg2rad(cfg.theta0_max_deg)) + 10.0)
        nr = int(round(z / 2.4))
    if cfg.po_n_azimuth > 0:
        na = cfg.po_n_azimuth
    else:
        z = (1.09 * (np.pi * cfg.D / cfg.lam)
             * np.sin(np.deg2rad(cfg.theta0_max_deg)) + 10.0)
        na = int(round(z))
    # 步长封顶（0 或负数表示不封顶）
    if verbose is None:
        verbose = getattr(cfg, 'verbose', True)
    s_max = getattr(cfg, 'po_max_step_lam', 0.5)
    if s_max and s_max > 0:
        nr_min = int(np.ceil((0.5 * cfg.D / cfg.lam) / s_max))
        na_min = int(np.ceil((np.pi * cfg.D / cfg.lam) / s_max))
        if (nr < nr_min or na < na_min) and verbose:
            print(f"  网格封顶：{nr}×{na} → {max(nr, nr_min)}×{max(na, na_min)}"
                  f"（方位步长原为 {(np.pi*cfg.D/na)/cfg.lam:.2f}λ，"
                  f"封到 ≤{s_max}λ 以免宽角出伪瓣）")
        nr = max(nr, nr_min)
        na = max(na, na_min)
    m = max(1, int(mult))
    nr = max(2, int(round(nr * m)))
    na = max(3, int(round(na * m)))
    d_r = (0.5 * cfg.D / nr) / cfg.lam          # 径向步长 / λ
    d_a = (np.pi * cfg.D / na) / cfg.lam        # 方位弧长 / λ（在口径圆心处）
    return nr, na, d_r, d_a


def compute_theta_0(D, F, h):
    """根据论文公式(2)计算反射器边缘半张角 θ0（弧度）"""
    term1 = (D + h) / (F - (D + h)**2 / (4 * F))
    term2 = h / (F - h**2 / (4 * F))
    theta_0 = 0.5 * (np.arctan(term1) - np.arctan(term2))
    return theta_0

def save_cut_file(file_path, theta_deg, gain_dbi, phi_deg=0.0):
    """
    保存方向图数据为标准 GRASP .cut 格式（单一切面）
    :param file_path: 输出文件名
    :param theta_deg: 角度数组（度，等间隔）
    :param gain_dbi: 增益数组（dBi）
    :param phi_deg: 切面角度（度）
    """
    # 确保角度为升序
    if theta_deg[0] > theta_deg[-1]:
        theta_deg = theta_deg[::-1]
        gain_dbi = gain_dbi[::-1]

    start_angle = theta_deg[0]
    step = theta_deg[1] - theta_deg[0]
    num_points = len(theta_deg)

    # 将增益转换为场强幅度（线性值）
    linear_gain = 10 ** (gain_dbi / 10.0)
    # 假设 Ex 和 Ey 各占一半功率，实部为幅度，虚部为 0
    e_mag = np.sqrt(linear_gain / 2.0)

    with open(file_path, 'w') as f:
        f.write("Field data in cuts\n")
        f.write(f"{start_angle:.6f} {step:.6f} {num_points} {phi_deg}\n")
        for mag in e_mag:
            f.write(f"{mag:.8e} 0.0 {mag:.8e} 0.0\n")


def _plot_gbma_vs_po(theta_deg, dbi_gbma, dbi_po, phi_deg=0.0, info=None,
                     path="gbma_vs_po.png"):
    """把 GBMA 与同一次运行内的 PO 参考画在一张图上，保存到 path"""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    pk = max(float(np.max(dbi_gbma)), float(np.max(dbi_po)))
    ax.plot(theta_deg, dbi_gbma - pk, "b-", lw=1.4, label="GBMA")
    ax.plot(theta_deg, dbi_po - pk, "r--", lw=1.1, label="PO (reference)")
    ttl = f"GBMA vs PO  (phi={phi_deg:g} deg)"
    if info:
        ttl += f"   N={info['n_beams']}"
    ax.set_title(ttl)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("relative level (dB)")
    ax.set_ylim(-60, 3)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  对比图已保存: {os.path.abspath(path)}")


# =============================================================================
# 参数配置：所有可调项都在这里
# =============================================================================
@dataclass
class AntennaConfig:
    # ---- 算法开关（想算什么就开哪个；GBMA 不依赖 PO）----
    run_po: bool = True         # 单点（形心）PO 数值积分 → PO.cut
    run_po_gauss: bool = False  # 多点 Gauss PO（本文件里尚未实现，仅占位）
    run_gbma: bool = False      # GBMA（馈源 GB 展开 → 逐束反射 → 合成）→ gbma.cut
    run_paper: bool = False     # 论文第三章法（实验）→ gbma_paper.cut
    run_gb_po: bool = False     # GB 展开 + 数值 PO（旧路线，保留占位）

    # ---- GBMA 参数（gbma.py）----
    #   馈源 → 一族高斯波束：六边形格点, 束数吸附到 1+3K(K+1)
    #   (7,19,37,61,91,127,169,217,271,331,...,547)
    gbma_n_beams: int = 91
    gbma_overlap: float = 1.2    # 格点角距 / GB 角半径；1.0~1.5 都可以
    gbma_theta_fill: float = 1.0  # 最外环角 = theta_fill * 口径张角（1.0 = 正好在边缘）
    gbma_reg: float = 1e-8       # 点匹配系数的 Tikhonov 正则
    gbma_phi_deg: float = 0.0    # GBMA 输出切面
    gbma_check_po: bool = True   # 顺带用 gbma.po_reference 算一条 PO 参考做对拍
    gbma_plot: bool = True       # 保存 gbma_vs_po.png 对比图
    # 合成用哪一支:
    #   'auto'    : |S_a|>1.5 用字面式, |S_a|<0.7 用稳定式, 中间 smoothstep 过渡 —— **默认**
    #   'stable'  : P_r * [1+erf(S_a)]/2  (74 的领先阶等价形式; 处处有限)
    #   'literal' : 字面式 T(S_a)*P_r + P_d (含更高阶端点项, 单束对拍里更准;
    #               但 S_a->0 时 T 与 P_d 各自按 1/S_a 发散、数值上不能完全相消)
    # 注意: S_a=0 只发生在"某束的镜面反射点正好落在反射面边缘"且"观察方向正好
    #       等于该束的镜面方向". 本工程正馈 + theta_fill=1.0 时, 外圈 30 束的束轴
    #       正好打在口径边缘, 而 theta=0 又是所有束的镜面方向 -> 只有 theta=0 这个点
    #       会炸(+600 dB); 其它角度 |S_a| 已 >=O(0.1), 字面式与稳定式一致.
    gbma_form: str = "auto"

    # 边缘点 phi1se 的解法（见 REPORT A.21）:
    #   'newton'       : 阻尼牛顿解精确 (B7) —— **默认**, 最稳
    #   'cubic'        : 三次解(B13近似)做起初值 + 牛顿（精度等价, 略快）
    #   'cubic_pure'   : 纯三次闭式, 无牛顿 —— 有 (B13) 前提, 宽角/正馈会崩(仅对照)
    #   'quartic_pure' : 纯四次闭式(无 B13、无迭代) —— 与牛顿同精度, 快 ~5x
    #   'closed'       : 四次闭式 + 坏点回退牛顿 —— 同一档速度, 多一层保险
    gbma_phi1se_mode: str = "newton"

    # 馈源分解方式（REPORT A.18 / A.23）:
    #   'near'     : 在**反射面上**做点匹配（默认；面上场误差 ~2%，端到端峰差 ~+0.2 dB）
    #   'farfield' : 在**馈源远场方向图**上做点匹配（chou2001 §II，见 gb_farfield_decomp.py）
    #                参数链 (19)→(15)→(13)，N 是产物；面上场误差 ~7-9%，
    #                但端到端峰差 0.05~0.27 dB、宽角 10-20° 反而略好
    gbma_decomp: str = "near"
    gbma_w_rel: float = 0.40        # 仅 farfield: (19) 窗口内选 w 的位置 [0,1]
    gbma_cone_margin: float = 1.10  # 仅 farfield: 覆盖锥相对边缘角的余量
    gbma_ff_basis: str = "paper"    # 仅 farfield: 'paper'=exp(-(kb/2)sin²)(1+cos)/2,
                                    #             'beam' =exp(-(tanγ/θgb)²)/cosγ

    # 边缘曲率（月牙）修正（REPORT A.24/A.25）——**默认关**
    #   True  + gbma_form="crescent" : T = ½[1+erf(S_a)] - S_ratio  ← 嫁接式，已不推荐
    #   **推荐改用下面的 form="analytic"**（纯解析独立路径，REPORT A.27）：
    #     同一分解下峰值偏差 0.151/0.215 → 0.102/0.130 dB，且反射阶段快 3~4×
    #     （它不再解 (B7) 的四次方程）。已知限制：θ≳20° 时宽角低 2~7 dB（见 A.27.4）。
    gbma_edge_crescent: bool = False
    gbma_crescent_order: int = 1    # 1 足够（实测 order=2 与 order=1 逐位相同）

    # ---- 仅 form="goedge"（GO 反射 + PO 边缘衍射，REPORT A.28）----
    # gbma_goedge_takenaka: 边缘项写成 Takenaka&Fukumitsu(1982) (25)+(26) 的分支形式
    #     T_st = E(Z) − (sgn/2)·erfc(sgn·Z)    （几何波 + 边界绕射波）
    #   False 则用代数恒等的紧致式 ½[1+erf(Z)]；实测相对差 4e-15 ⟹ **数值完全一致**，
    #   只影响代码写法（True 更贴近论文）。
    # gbma_goedge_crescent: 是否再叠加「曲边/月牙」修正 S_ratio。
    #   Takenaka 的典范解是**直线半屏**，我们的边缘是椭圆；实测加了更好
    #   （12 GHz 峰值偏差：正馈 near 0.148→0.099、偏馈 farfield 0.193→0.030；
    #     全段 RMS 1.90→0.99）⟹ 默认 True。
    gbma_goedge_takenaka: bool = True
    gbma_goedge_crescent: bool = True

    # ---- 额外输出曲线（同一分解/同一馈源功率，另存 .cut）----
    #   'go'      : 只考虑反射、不做边缘修正 = 论文 L750-753 的 P_r
    #               （"若 (64) 的鞍点在口径内，则它就是**边缘不存在时**的反射场"）
    #   'go2'     : 只考虑反射、不做边缘修正，但用 goedge 的 **GO 反射**写法
    #               （冻结矢量幅度 F0 + 解析远场）= Rieckmann 2002 的"几何波"那一半
    #   'literal' : 字面式 (74) T*P_r + P_d（含端点项；|S_a|→0 处有奇性，已内部兜底）
    #   'goedge'  : GO 反射 + PO 边缘衍射（Takenaka erfc 分支式 + 月牙，REPORT A.28）
    #   'goedge2' : 同上，但用**扁平化**实现（GoEdgeFast），快 ~2×（REPORT A.32）
    #   'stable' / 'auto' / 'analytic' 也都可以放进来
    #   每项会在 c.py 的汇总表里多一行，并多写一个 <form>.cut
    #   注意：'go'/'go2'/'literal' 等与主曲线共用同一次 Ps（边际 ~2 ms）；
    #         'go2'/'goedge2'/'literal2' 是独立批量调用（~30/~60/~320 ms）；
    #         'goedge'/'analytic' 各有解器。
    #   'literal2' = 扁平化的 literal（数值与 literal 等价，实测 8e-16），见 REPORT A.38
    gbma_extra_forms: tuple = ("go", "go2", "literal", "goedge", "goedge2", "literal2")

    # ---- 天线几何 / 电参数 ----
    freq: float = 20e9
    F: float = 0.6
    D: float = 1.0
    offset_x: float = 0.0       # 正馈：投影圆心位于 (0.0, 0, 0)
    offset_y: float = 0.0
    edge_angle_deg: float = 45.0
    edge_taper_db: float = -12.0
    theta0_max_deg: float = 30.0    # PO 网格自动公式用的扫描上限
    theta_scan_deg: tuple = (0, 20, 361)   # (起始°, 终止°, 点数)

    # ---- PO 数值积分网格 ----
    # 0 = 用自动公式（由 D/λ 与 theta0_max_deg 决定）；>0 = 直接用这个数。
    po_n_radial: int = 0
    po_n_azimuth: int = 0
    # 单元步长上限（以波长计，0=不封顶）。自动公式只按扫描范围定网格，
    # 方位常到 2.4λ，会在 ~24° 造出 +30 dB 的伪散射瓣 → 必须封顶。
    # 0.5λ 收敛；1.0λ 尚可；大于 1.5λ 宽角不可信。
    po_max_step_lam: float = 0.5
    # 多点 Gauss PO：求积规则 × 网格加密倍数
    #   'gl2x2'/'gl3x3'/'gl4x4' —— 极坐标张量积 Gauss–Legendre（谱收敛）
    #   '1'/'3'/'7'/'13'        —— 三角形 Dunavant 规则（固定代数精度）
    po_gauss_rule: str = 'gl3x3'
    po_gauss_mult: int = 1

    # 说明: GBMA 展开参数就是 gbma_n_beams / gbma_overlap / gbma_theta_fill / gbma_reg;
    # 边缘处理 = chou1997 (74), 合成可选 stable / literal / auto (gbma_form)。
    # 早期那套 (mode/spot_fill/edge_order/edge_method...) 已删除。


    # ---- 输出开关 ----
    # True  : 打印全部诊断信息（网格、LC、光斑/反射面、设置…）
    # False : 只打印每个算法的"峰值 dBi + 耗时 s"一行；库内部的所有
    #         print/warning 都被重定向丢弃，计时不受 I/O 干扰。
    verbose: bool = True

    @property
    def lam(self):
        return 3e8 / self.freq

    @property
    def k0(self):
        return 2 * np.pi / self.lam


# =============================================================================
# 主流程：按开关执行所选算法并保存 .cut
# =============================================================================
def _run_impl(cfg: AntennaConfig, report: list):
    """按 cfg.run_* 开关执行所选算法，结果存成 GRASP .cut。

    GBMA 只依赖馈源模型，不依赖 PO 网格；关掉 PO 时可省掉网格生成。
    """
    t_all = time.perf_counter()
    # 先校验 edge_method：避免跑完几分钟网格/P0 才在 GBMA 里报错，
    # 也避免旧版那种"未知字符串静默降级成 series"的坑（见 gbma_edge 注释）。
    # cfg.gbma_edge_method = normalize_edge_method(cfg.gbma_edge_method)
    feed = FeedAntenna(cfg.freq, edge_angle_deg=cfg.edge_angle_deg,
                       edge_taper_db=cfg.edge_taper_db)
    P_feed_total = compute_feed_total_power(feed)

    # 馈源指向（从焦点指向反射面中心）
    focus = np.array([0.0, 0.0, cfg.F])
    z_center = (cfg.offset_x**2 + cfg.offset_y**2) / (4 * cfg.F)
    feed_axis = np.array([cfg.offset_x, cfg.offset_y, z_center]) - focus
    feed_axis /= np.linalg.norm(feed_axis)

    # 扫描角度
    theta_start, theta_stop, num_theta = cfg.theta_scan_deg
    theta_scan = np.linspace(theta_start, theta_stop, num_theta)

    print(f"配置: f={cfg.freq/1e9:g} GHz, D={cfg.D:g} m (D/λ={cfg.D/feed.lam:.1f}), "
          f"F={cfg.F:g} m (F/D={cfg.F/cfg.D:g}), 偏移=({cfg.offset_x:g},{cfg.offset_y:g})")
    print(f"      扫描 {theta_start:g}°~{theta_stop:g}°, {num_theta} 点;  "
          f"算法: " + ", ".join(
              n for n, on in (('PO', cfg.run_po), ('PO-Gauss', cfg.run_po_gauss),
                              ('GBMA', cfg.run_gbma),
                              ('GB-PO', cfg.run_gb_po),
                              ('论文法', cfg.run_paper)) if on))

    # ---- 只在需要时生成反射面网格 ----
    tri_verts = tri_norms = tri_areas = None
    if cfg.run_po or cfg.run_po_gauss:
        nr, na, d_r, d_a = po_mesh_sizes(cfg, 1)
        tri_verts, tri_norms, tri_areas = generate_parabolic_mesh(
            cfg.F, cfg.D, N_radial=nr, N_azimuth=na,
            offset_x=cfg.offset_x, offset_y=cfg.offset_y)
        print(f"网格: N_radial={nr}, N_azimuth={na}  "
              f"(径向 {d_r:.2f}λ, 方位 {d_a:.2f}λ)")
        if d_a > 2.0 * d_r:
            print(f"  提示：方位 {d_a:.2f}λ 比径向 {d_r:.2f}λ 粗 "
                  f"{d_a/d_r:.1f} 倍，方位是限制方向。")

    saved = []

    # ---------- PO（单点形心）----------
    if cfg.run_po:
        print("计算 PO 常数近似 ...")
        t0 = time.perf_counter()
        positions_c = np.mean(tri_verts, axis=1)
        norms_c = np.mean(tri_norms, axis=1)
        norms_c /= np.linalg.norm(norms_c, axis=1, keepdims=True)
        H_inc_c = compute_incident_H(positions_c, feed, cfg.F, feed_axis)
        po_c = POIntegrator(positions_c, norms_c, tri_areas, feed.k)
        E_c = po_c.far_field_pattern(H_inc_c, theta_scan, phi_deg=0.0)
        U_c = np.sum(np.abs(E_c)**2, axis=-1)
        dBi_po = 10 * np.log10(4 * np.pi * U_c / P_feed_total + 1e-15)
        dt = time.perf_counter() - t0
        report.append(("PO", float(np.max(dBi_po)), dt))
        print(f"PO 峰值方向性: {np.max(dBi_po):.2f} dBi, 耗时 {dt:.2f}s")
        save_cut_file("PO.cut", theta_scan, dBi_po, phi_deg=0.0)
        saved.append("PO.cut")


    # ---------- GBMA：馈源 GB 展开 → 逐束反射 → 远场合成 ----------
    if cfg.run_gbma:
        print("计算 GBMA（馈源 GB 展开 → 逐束反射 → 合成）...")
        t0 = time.perf_counter()
        try:
            E_gbma, dBi_gbma, info = compute_far_field_gbma(
                feed, cfg, theta_scan,
                n_beams=cfg.gbma_n_beams,
                overlap=cfg.gbma_overlap,
                theta_fill=cfg.gbma_theta_fill,
                reg=cfg.gbma_reg,
                verbose=cfg.verbose,
                phi_deg=cfg.gbma_phi_deg,
                form=getattr(cfg, "gbma_form", "auto"),
                phi1se_mode=getattr(cfg, "gbma_phi1se_mode", "newton"),
                decomp=getattr(cfg, "gbma_decomp", "near"),
                w_rel=getattr(cfg, "gbma_w_rel", 0.40),
                cone_margin=getattr(cfg, "gbma_cone_margin", 1.10),
                ff_basis=getattr(cfg, "gbma_ff_basis", "paper"),
                edge_crescent=getattr(cfg, "gbma_edge_crescent", False),
                crescent_order=getattr(cfg, "gbma_crescent_order", 1),
                goedge_takenaka=getattr(cfg, "gbma_goedge_takenaka", True),
                goedge_crescent=getattr(cfg, "gbma_goedge_crescent", True),
                extra_forms=getattr(cfg, "gbma_extra_forms", ()),
            )
            dt = time.perf_counter() - t0
        except Exception as exc:
            print(f"  GBMA 失败（已跳过）: {type(exc).__name__}: {exc}")
            E_gbma = dBi_gbma = info = None
        if dBi_gbma is not None:
            if not np.all(np.isfinite(dBi_gbma)):
                print("  警告: GBMA 返回非有限值，跳过该曲线")
            else:
                report.append(("GBMA", float(np.max(dBi_gbma)), dt))
                print(f"GBMA 峰值方向性: {np.max(dBi_gbma):.2f} dBi  ({info['n_beams']} 束, "
                      f"合成式={info.get('form', '?')}, "
                      f"phi1se={info.get('phi1se_mode', '?')}, "
                      f"分解={info.get('decomp', '?')}, 耗时 {dt:.2f}s)")
                if getattr(cfg, "gbma_edge_crescent", False) and \
                        str(getattr(cfg, "gbma_form", "auto")).lower() != "crescent":
                    print("  注意: gbma_edge_crescent=True 但 gbma_form != 'crescent' —— "
                          "本次**没有**用上月牙修正（Ps_crescent 需要 form='crescent'）")
                if str(getattr(cfg, "gbma_form", "auto")).lower() == "goedge":
                    print(f"  边缘项: Takenaka(25)+(26) "
                          f"{'✓' if info.get('goedge_takenaka', True) else '✗ 用紧致 erf 式'}，"
                          f"月牙 {'✓' if info.get('goedge_crescent', True) else '✗'}")
                save_cut_file("gbma.cut", theta_scan, dBi_gbma, phi_deg=cfg.gbma_phi_deg)
                saved.append("gbma.cut")
                # 若主曲线本身就在额外列表里（如 form="literal"），也按它的名字另存一份，
                # 这样 go.cut / literal.cut 一定都会出现
                _main_f = str(getattr(cfg, "gbma_form", "auto")).lower()
                _want = {str(f).lower() for f in (getattr(cfg, "gbma_extra_forms", ()) or ())}
                if _main_f in _want:
                    _alias = f"{_main_f}.cut"
                    save_cut_file(_alias, theta_scan, dBi_gbma, phi_deg=cfg.gbma_phi_deg)
                    if _alias not in saved:
                        saved.append(_alias)
                # 额外合成式：**已在上面的同一次遍历里算好**（REPORT A.30），这里只存盘
                _ex = info.get("extra_dBi") or {}
                _t_by = info.get("t_acc_by_form") or {}
                for _f, dBi_x in _ex.items():
                    if not np.all(np.isfinite(dBi_x)):
                        print(f"  额外曲线 form={_f} 含非有限值（已跳过）")
                        continue
                    _name = f"{_f}.cut"
                    save_cut_file(_name, theta_scan, dBi_x, phi_deg=cfg.gbma_phi_deg)
                    saved.append(_name)
                    # 各形式**实测**的边际成本：共享 Ps 的（stable/literal/auto/go/...）
                    # 只有"旋转+活塞+累加"；go2 是独立批量调用，整段都算它的。
                    _t_m = float(_t_by.get(_f, 0.0))
                    report.append((_f.upper(), float(np.max(dBi_x)), _t_m))
                    print(f"  额外曲线 {_f}: → {_name}  (峰值 {np.max(dBi_x):.2f} dBi, "
                          f"边际 {_t_m*1e3:.1f} ms)")
                # 用同一馈源模型的数值 PO 做对拍（可选）
                if cfg.gbma_check_po:
                    S_chk = directions(theta_scan, cfg.gbma_phi_deg)
                    E_ref = po_reference(info["expansion"], S_chk)
                    U_ref = np.sum(np.abs(E_ref) ** 2, axis=-1)
                    dBi_ref = 10.0 * np.log10(4.0 * np.pi * U_ref / P_feed_total + 1e-30)
                    msk = dBi_ref > (np.max(dBi_ref) - 60.0)
                    rms = float(np.sqrt(np.mean((dBi_gbma - dBi_ref)[msk] ** 2)))
                    print(f"  对拍: 峰值差 {np.max(dBi_gbma)-np.max(dBi_ref):+.2f} dB, "
                          f"RMS(>−60dB) {rms:.2f} dB")
                    if cfg.gbma_plot:
                        _plot_gbma_vs_po(theta_scan, dBi_gbma, dBi_ref,
                                         cfg.gbma_phi_deg, info)


def run_analysis(cfg: AntennaConfig):
    """按 cfg.run_* 开关执行所选算法并保存 .cut。

    cfg.verbose=False 时把 _run_impl 里所有 print 重定向丢弃（库内部
    的 print/warning 一并静默），只输出一张"算法 / 峰值 / 耗时"表 ——
    这样计时不受任何 I/O 干扰。
    """
    report = []
    if cfg.verbose:
        _run_impl(cfg, report)
    else:
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            _run_impl(cfg, report)

    if report:
        print(f"{'算法':<12}{'峰值(dBi)':>12}{'耗时(s)':>12}")
        for name, peak, dt in report:
            print(f"{name:<12}{peak:>12.2f}{dt:>12.3f}")



# 向后兼容旧名字
compute_and_save_po_gbma = run_analysis


# =============================================================================
# 主程序：所有配置都在这里改
# =============================================================================
if __name__ == "__main__":
    config = AntennaConfig()

    # ---------- 1. 算法开关 ----------
    config.run_po = True        # 单点 PO（数值积分）→ PO.cut
    config.run_po = False       # 单点 PO（数值积分）→ PO.cut

    config.run_po_gauss = False   # 多点 Gauss PO（未实现，占位）
    config.run_gbma = True        # GBMA（馈源 GB 展开 → 逐束反射 → 合成）→ gbma.cut
    config.run_gb_po = False      # GB 展开 + 数值 PO（旧路线，占位）
    config.run_paper = False      # 论文第三章法（未实现，占位）

    # ---------- 2. 几何 / 电参数 ----------
    config.freq = 50e9            # 12 GHz
    config.D = 1.0                # 口径 1 m
    config.F = 0.6                # 焦距 0.6 m（F/D = 0.6）
    config.offset_x = 0.5# 正馈
    config.offset_y = 0.0
    config.edge_angle_deg = 45.0
    config.edge_taper_db = -12.0

    # ---------- 3. 扫描角度 ----------
    config.theta_scan_deg = (0, 40, 1601)
    config.theta0_max_deg = 50.0  # PO 网格自动公式的扫描上限

    # ---------- 4. PO 数值积分网格 ----------
    config.po_max_step_lam = 0.5  # 步长上限（λ）。必须 ≤0.5，否则宽角出伪瓣
    config.po_n_radial = 0        # 0 = 自动
    config.po_n_azimuth = 0       # 0 = 自动
    config.po_gauss_rule = 'gl3x3'
    config.po_gauss_mult = 1

    # ---------- 5. GBMA 配置（见 gbma.py 顶部说明）----------
    # 链路: 馈源分解(六边形格点 GB, 点匹配求系数) -> 逐束 chou1997 闭式反射
    #        -> 统一相位参考点后相干合成
    # 束数会吸附到最近的 1+3K(K+1): 7,19,37,61,91,127,169,217,271,331,...,547
    #   实测(12 GHz/D=1/F=0.6, ±20°/361 点, 对同馈源 PO):
    #     91 束  -> 峰值差 −0.12 dB, RMS 0.9~1.3 dB, 耗时 ~2 s
    #     37 束  -> 主瓣略差, 宽角 ~2 dB
    config.gbma_n_beams = 91
    config.gbma_overlap = 1.2      # 格点角距/GB 角半径: 1.0~1.5
    config.gbma_theta_fill = 1.0   # 最外环角 = 1.0 * 口径张角(正好落在边缘)
    config.gbma_reg = 1e-8         # 点匹配系数的 Tikhonov 正则
    config.gbma_phi_deg = 0.0      # GBMA 输出切面
    config.gbma_check_po = False# 顺带算一条同馈源数值 PO 参考并对拍
    config.gbma_plot = True        # 保存 gbma_vs_po.png
    # 合成式：只允许一行生效（stable 默认；literal 仅作对照，且 12 GHz 实测更差）
    config.gbma_form = "stable"    # 默认，生产用
    config.gbma_form = "literal"   # 字面 (74)，仅作对照；会额外打印告警
    # config.gbma_form = "goedge"    # 默认，生产用

    # 边缘点 phi1se 解法（REPORT A.21）：只允许一行生效
    #   newton       : 阻尼牛顿解精确 (B7) —— 最稳
    #   quartic_pure : 解 (B7) 的四次方程（cos=±√(1-sin²) 精确代入，**不做**论文 (B13) 的
    #                  cos≈1-sin²/2 小角近似）=> 精度与 newton 等同(方向图差 ≤0.05 dB),
    #                  反射阶段快约 5×
    #   closed       : 同上 + 坏点回退牛顿（更保险）
    #   注: cubic_pure 是 (B13) 的**三次近似**版, 要求 sin(phi1se)<1/4; 本工程宽角处
    #       55~70% 的 (束,方向) 不满足, 正馈会崩(+254 dB), 仅供对照
    # config.gbma_phi1se_mode = "newton"        # 默认：阻尼牛顿解 (B7)
    config.gbma_phi1se_mode = "quartic_pure"  # 四次闭式(无 B13)，精度=牛顿，快约 5×
    # config.gbma_phi1se_mode = "closed"        # 四次闭式 + 坏点回退牛顿
    # config.gbma_phi1se_mode = "cubic_pure"    # (B13) 三次近似，仅对照

    # 馈源分解方式（REPORT A.18 / A.23）：只允许一行生效
    #   near     : 反射面上点匹配（默认）—— 面上场误差 ~2%，端到端峰差 +0.15~0.23 dB
    #   farfield : 馈源远场方向图点匹配（chou2001 §II，gb_farfield_decomp.py）
    #              N 由 (19)/(15)/(13) 决定（本配置 ≈37），面上场误差 7~9%，
    #              但端到端峰差 0.05~0.27 dB、10-20° 宽角反而略好
    # config.gbma_decomp = "near"
    config.gbma_decomp = "farfield"
    config.gbma_w_rel = 0.40          # 仅 farfield：(19) 窗口内位置 [0,1]，0=下界
    config.gbma_cone_margin = 1.10    # 仅 farfield：覆盖锥相对边缘角的余量
    config.gbma_ff_basis = "paper"    # 仅 farfield：'paper' / 'beam'

    config.verbose = False   # True: 完整诊断；False: 只出峰值/耗时表

    # 边缘曲率（月牙）修正（REPORT A.24/A.25）：**嫁接式，已不推荐**
    #   edge_crescent=True 只是让 Ps 多算一个 Ps_crescent；真正用上它还要 form="crescent"
    # config.gbma_edge_crescent = True
    # config.gbma_form = "crescent"

    # 纯解析路径（REPORT A.27）：反射本体(二次型 SPP) + 直边 erf + 月牙级数，
    #   **不含任何复根求解**（不再解 (B7) 的四次方程）
    #   实测（θ∈[0,40]/361）：反射阶段 515/580/390/566 ms → 188/144/126/120 ms（3~4×）
    #   峰值偏差全面变小：正馈near 0.151→0.102，正馈farfield 0.069→0.047，
    #                    偏馈near 0.215→0.130，偏馈farfield 0.195→0.033
    #   已知限制：farfield 分解在 θ≳20° 比 PO 低 2~7 dB（宽角端点项缺失，A.27.4）
    # config.gbma_form = "analytic"

    # GO 反射 + PO 边缘衍射（REPORT A.28）：最快、主瓣/峰值最好
    #   反射阶段 67~83 ms（stable 371~393）⟹ 约 5×
    #   峰值偏差：+0.099 / −0.053 / +0.130 / +0.030（四构型）
    #   边缘项 = Takenaka&Fukumitsu(1982) (25)+(26) 的"几何波 E(Z) + 绕射波 erfc"
    #     两个开关：gbma_goedge_takenaka（写法，数值不变）、gbma_goedge_crescent（曲率项）
    #   已知限制：farfield/偏馈 near 的 10–40° 仍不如 stable（1.65/3.70 vs 0.40/2.44）
    # config.gbma_form = "goedge"
    # config.gbma_goedge_takenaka = True    # False = 紧致 erf 式（数值相同）
    # config.gbma_goedge_crescent = False   # False = 纯 Takenaka（无曲边修正）

    run_analysis(config)
