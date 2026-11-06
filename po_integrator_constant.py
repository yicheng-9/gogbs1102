import numpy as np

class POIntegrator:
    def __init__(self, positions, normals, areas, k):
        self.positions = np.asarray(positions, dtype=float)
        self.normals   = np.asarray(normals, dtype=float)
        self.areas     = np.asarray(areas, dtype=float)
        self.k         = float(k)

    def far_field_pattern(self, H_inc, theta_deg, phi_deg=0.0, chunk=None):
        """PO 远场方向图（常数近似）。

        chunk: 三角形分块大小（None=按内存上限自动分块）。分块只影响内存
        占用，不影响结果（对三角形求和是精确的分块累加）。
        """
        theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
        if np.isscalar(phi_deg):
            phi = np.full_like(theta, np.deg2rad(float(phi_deg)))
        else:
            phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
            if phi.shape != theta.shape:
                raise ValueError("phi_deg 必须与 theta_deg 同形状")

        N = theta.size
        sin_t = np.sin(theta)
        S = np.empty((N, 3), dtype=float)
        S[:, 0] = sin_t * np.cos(phi)
        S[:, 1] = sin_t * np.sin(phi)
        S[:, 2] = np.cos(theta)

        S_T = S.T                                   # (3, N)
        Js_all = 2.0 * np.cross(self.normals, H_inc)  # (M, 3)
        M = Js_all.shape[0]

        if chunk is None:
            # 目标：单块 (chunk, N, 3) 复数组 <= 约 200 MB
            bytes_per_tri = N * 3 * 16
            chunk = max(1, int(2.0e8 // max(bytes_per_tri, 1)))
            chunk = min(chunk, M)

        integral = np.zeros((N, 3), dtype=complex)
        for i0 in range(0, M, chunk):
            i1 = min(i0 + chunk, M)
            dot_n_s = self.normals[i0:i1] @ S_T            # (c, N)
            mask = dot_n_s > 0
            Js = Js_all[i0:i1]                             # (c, 3)
            s_dot_J = Js @ S_T                             # (c, N)
            term = S[None, :, :] * s_dot_J[:, :, None] - Js[:, None, :]
            phase = np.exp(1j * self.k * (self.positions[i0:i1] @ S_T))
            part = term * (phase[:, :, None]
                           * self.areas[i0:i1, None, None])
            part[~mask] = 0.0
            integral += np.sum(part, axis=0)

        Es = -1j * self.k / (4 * np.pi) * integral
        return Es

    def rcs_from_Es(self, Es):
        return np.sum(np.abs(Es)**2, axis=-1)