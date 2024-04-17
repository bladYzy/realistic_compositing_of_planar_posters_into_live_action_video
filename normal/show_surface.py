import cv2
import numpy as np
from sklearn.cluster import KMeans

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm else v

def process_normal_map(normal_map):
    # 将RGB值从[0, 255]映射到[-1, 1]
    normals = (normal_map.astype(np.float32) / 127.5) - 1
    # 归一化法线向量
    normals = np.apply_along_axis(normalize_vector, 2, normals)
    return normals.reshape(-1, 3)

def cluster_planes(normals, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(normals)
    return labels.reshape(normal_map.shape[0], normal_map.shape[1])

def color_planes(image, labels):
    unique_labels = np.unique(labels)
    colored_image = np.zeros_like(image)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in unique_labels]
    for label, color in zip(unique_labels, colors):
        colored_image[labels == label] = color
    return colored_image

normal_map = cv2.imread('test9_frame_normal_resize.png')
image = cv2.imread('test9_frame_og.jpg')

# 处理法线图并聚类平面
normals = process_normal_map(normal_map)
labels = cluster_planes(normals)

# 为不同平面上色
colored_image = color_planes(image, labels)
cv2.imwrite("visual_planes.png", colored_image)
# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Colored Planes', colored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

