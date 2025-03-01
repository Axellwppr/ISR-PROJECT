import numpy as np
from utils import plot_dynamic_points
from scipy.spatial.transform import Rotation as sRot

def apply_ema(data, alpha=0.4):
    """
    对数据进行指数移动平均处理
    alpha: 平滑因子，范围(0,1)，值越小平滑效果越强
    """
    result = np.zeros_like(data)
    result[0] = data[0]
    for t in range(1, len(data)):
        result[t] = alpha * data[t] + (1 - alpha) * result[t-1]
    return result

data = np.load("T-omni_joints.npy")

# 对数据进行EMA处理
data = apply_ema(data)

rot2 = sRot.from_euler('xz', [-100,0], degrees=True)

print(rot2.as_rotvec())

root = data[:, 0:1, :]
data = rot2.apply((data - root).reshape(-1, 3)).reshape(-1, 9, 3)

data[:,:,1] =0 

# plot_dynamic_points(data, data)
print(data[0])

np.save("omni_joints_rotated.npy", data)