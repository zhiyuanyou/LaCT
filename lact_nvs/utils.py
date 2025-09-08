import numpy as np
import torch
from sklearn.decomposition import PCA


def cal_ground_normal(points):
    points = np.array(points)
    points_centered = points - points.mean(axis=0)
    points_centered = np.array(points_centered)
    # PCA 分解
    pca = PCA(n_components=3)
    pca.fit(points)
    # 最小特征值对应的分量方向
    normal = pca.components_[-1]
    normal = normal / np.linalg.norm(normal)
    # 拟合度 R^2
    lambdas = pca.explained_variance_
    variance_ratio = lambdas[-1] / sum(lambdas)
    r2 = 1 - variance_ratio
    # 点到平面距离的 RMSE
    dists = np.dot(points_centered, normal)
    rmse = np.sqrt(np.mean(dists**2))
    return normal, r2, rmse


def rectify_c2w(c2w: torch.Tensor, world_up: torch.Tensor) -> torch.Tensor:
    """
    c2w: [4, 4] camera-to-world
    world_up: [3] 平面法向量
    返回矫正后的 c2w, 保持相机中心不变
    """
    def normalize(v, eps=1e-9):
        return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

    R = c2w[:3, :3]
    t = c2w[:3, 3]

    r = normalize(R[:, 0])
    u = normalize(R[:, 1])
    f = normalize(R[:, 2])

    up = normalize(world_up)

    # world_up 在与 f 正交平面的投影（只在该平面内改 roll）
    up_proj = up - (up * f).sum(dim=-1, keepdim=True) * f
    if up_proj.norm() < 1e-7:          # 退化：f ≈ ± up 时，用旧 u 兜底
        up_proj = u
    up_proj = normalize(up_proj)

    # 判定“接近水平”还是“接近竖直”
    is_landscape = torch.abs((u * up).sum()) >= torch.abs((r * up).sum())

    if is_landscape:
        # 水平：roll → 0° 或 180°（让 u_new 与 up_proj 同/反向，取更接近原 u 的那一侧）
        s = 1.0 if (u * up_proj).sum() >= 0 else -1.0
        u_new = s * up_proj
        r_new = normalize(torch.cross(u_new, f, dim=-1))
        f_new = f  # f 保持不变
    else:
        # 竖直：roll → ±90°（让 r_new 与 up_proj 同/反向，取更接近原 r 的那一侧）
        s = 1.0 if (r * up_proj).sum() >= 0 else -1.0
        r_new = normalize(s * up_proj)
        u_new = normalize(torch.cross(f, r_new, dim=-1))
        f_new = f  # f 保持不变

    R_new = torch.stack([r_new, u_new, f_new], dim=-1)

    out = c2w.clone()
    out[:3, :3] = R_new
    out[:3, 3]  = t
    out[3, :]   = torch.tensor([0, 0, 0, 1], dtype=out.dtype, device=out.device)
    return out


def Rx(theta): c,s=np.cos(theta),np.sin(theta); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(theta): c,s=np.cos(theta),np.sin(theta); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(theta): c,s=np.cos(theta),np.sin(theta); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def rotate_view_c2w(c2w, yaw_deg=0, pitch_deg=0, roll_deg=0):
    c2w = c2w.numpy()
    # 约定：yaw 绕相机 y 轴，pitch 绕相机 x 轴，roll 绕相机 z 轴
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    R_cam_delta = Rz(roll) @ Ry(yaw) @ Rx(pitch)   # 顺序可按需调整
    R = c2w[:3, :3].astype(float)
    t = c2w[:3, 3].astype(float)

    # 相机系内旋作用在右侧，取转置
    R_new = R @ R_cam_delta.T

    # 可选：数值正交化，抑制漂移
    U, _, Vt = np.linalg.svd(R_new)
    R_new = U @ Vt

    out = np.eye(4)
    out[:3, :3] = R_new
    out[:3, 3] = t  # 位置不变
    return torch.tensor(out, dtype=torch.float32)
