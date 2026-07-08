import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================= 配置路径 =================
JSON_PATH = r"D:\Project\LF_dataset\Calibration\Output\calibration_raw_stereo_locked_secondsyn.json"
# ===========================================

def plot_array():
    if not os.path.exists(JSON_PATH):
        print(f"错误：找不到文件 {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # 提取坐标
    cams = sorted(data.keys())
    x = [data[cam]['T_rel'][0] for cam in cams]
    y = [data[cam]['T_rel'][1] for cam in cams]
    z = [data[cam]['T_rel'][2] for cam in cams]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 画出相机位置点
    # 用颜色深浅代表 Z 轴深度偏移，更直观
    sc = ax.scatter(x, y, z, c=z, cmap='coolwarm', s=100, edgecolors='k', alpha=0.8)
    
    # 2. 添加机位标签 (A1, B2...)
    for i, cam in enumerate(cams):
        ax.text(x[i], y[i], z[i] + 0.5, cam, fontsize=9, ha='center')

    # 3. 绘制连接线（画出 5x5 网格感）
    # 按行连接
    rows = ['A', 'B', 'C', 'D', 'E']
    for r in rows:
        r_cams = [c for c in cams if c.startswith(f'CAM_{r}')]
        if len(r_cams) > 1:
            rx = [data[c]['T_rel'][0] for c in r_cams]
            ry = [data[c]['T_rel'][1] for c in r_cams]
            rz = [data[c]['T_rel'][2] for c in r_cams]
            ax.plot(rx, ry, rz, 'gray', alpha=0.3)

    # 按列连接
    for col in range(1, 6):
        c_cams = [c for c in cams if c.endswith(str(col))]
        if len(c_cams) > 1:
            cx = [data[c]['T_rel'][0] for c in c_cams]
            cy = [data[c]['T_rel'][1] for c in c_cams]
            cz = [data[c]['T_rel'][2] for c in c_cams]
            ax.plot(cx, cy, cz, 'gray', alpha=0.3)

    # 4. 设置轴标签
    ax.set_xlabel('X (mm) - Width')
    ax.set_ylabel('Y (mm) - Height')
    ax.set_zlabel('Z (mm) - Depth')
    ax.set_title('5x5 Light Field Camera Array Reconstruction')

    # 5. 保持纵横比一致，防止视觉变形（核心步骤）
    # Matplotlib 3.3.0+ 支持 set_box_aspect
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((np.ptp(x), np.ptp(y), 50)) # 强制 Z 轴显示范围，突出微小偏差

    plt.colorbar(sc, label='Z-axis Deviation (mm)', pad=0.1)
    print("3D 可视化窗口已开启。")
    plt.show()

if __name__ == "__main__":
    import os
    plot_array()