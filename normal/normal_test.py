import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

depth_map = cv2.imread('test1.jpg', cv2.IMREAD_UNCHANGED)

# 将深度图转换为点云
def depth_to_pointcloud(depth_map):
    # 这里需要根据你的深度相机的具体参数来调整
    h, w = depth_map.shape
    fx, fy = 1.0, 1.0  # 假设的焦距
    cx, cy = w / 2, h / 2
    points = []
    for y in range(h):
        for x in range(w):
            Z = depth_map[y, x]
            if Z > 0:  # 忽略深度为0的点
                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                points.append([X, Y, Z])
    return np.array(points)

points = depth_to_pointcloud(depth_map)

# 使用RANSAC算法检测平面
model, inliers = ransac(points, FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=1000)

selected_points = points[inliers]

# 选中的平面点可以用来估算平面的法向量
# 在这个简单的示例中，我们假设选中的平面较为平整，直接计算其法向量
# 更复杂的场景可能需要更精细的计算方法
mean_normal = np.mean(np.cross(selected_points[:-1] - selected_points[1:], selected_points[1:] - selected_points[2:]), axis=0)
mean_normal = mean_normal / np.linalg.norm(mean_normal)

print("Estimated plane normal vector:", mean_normal)
