import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def select_area_and_visualize_normal(frame_path, normal_map_path):
    frame = Image.open(frame_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    plt.title('Click two points to select an area')

    points = plt.ginput(2)
    plt.close()

    normal_map = np.array(Image.open(normal_map_path))

    # 计算边界
    x1, y1 = int(min(points[0][0], points[1][0])), int(min(points[0][1], points[1][1]))
    x2, y2 = int(max(points[0][0], points[1][0])), int(max(points[0][1], points[1][1]))

    # 计算平均法线
    selected_normals = normal_map[y1:y2, x1:x2]
    average_normal = np.mean(selected_normals, axis=(0, 1))

    norm = np.linalg.norm(average_normal)
    normalized_normal = average_normal / norm if norm > 0 else average_normal

    print(f"Average Normal: {average_normal}")
    print(f"Normalized Normal Vector: {normalized_normal}")



frame_path = 'test9_frame.jpg'
normal_map_path = 'test9_frame_normal.png'
select_area_and_visualize_normal(frame_path, normal_map_path)
