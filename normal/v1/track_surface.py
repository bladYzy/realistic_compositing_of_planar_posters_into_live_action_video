import cv2
import numpy as np
from sklearn.cluster import KMeans

def main(video_path, normal_map_path, poster_path):
    # 加载视频和图片
    cap = cv2.VideoCapture(video_path)
    normal_map = cv2.imread(normal_map_path)
    poster = cv2.imread(poster_path)

    # 第一帧处理
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频")
        return

    # 平面区域识别
    plane_areas = detect_planes(normal_map)

    # 用户选择区域
    selected_region = select_region(frame)

    # 初始化追踪器
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, selected_region)

    # 处理视频的每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 追踪区域
        success, box = tracker.update(frame)
        if success:
            fill_color(frame, box, plane_areas)

        # 显示结果
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect_planes(normal_map):
    # 将图像数据转换为 (n_samples, n_features) 形式的数组
    # 对于法线图，每个像素的RGB值是其特征
    data = normal_map.reshape((-1, 3))

    # 使用 K-means 算法来聚类，假设我们有5个平面区域
    kmeans = KMeans(n_clusters=5, random_state=0).fit(data)

    # 将聚类标签转换回原始图像尺寸，以便每个像素都有一个聚类标签
    labels = kmeans.labels_.reshape(normal_map.shape[:2])

    # 创建一个颜色映射，将每个聚类标签映射到一种颜色
    label_colormap = np.random.randint(0, 255, size=(kmeans.n_clusters, 3))

    # 根据聚类标签为每个像素分配颜色
    segmented_image = label_colormap[labels]

    # 将分段图像转换为8位无符号整数类型，以便显示和处理
    segmented_image = segmented_image.astype(np.uint8)

    return segmented_image


def select_region(frame):
    roi = cv2.selectROI("Select Region", frame, showCrosshair=True, fromCenter=False)

    # 关闭选择窗口
    cv2.destroyWindow("Select Region")

    return roi

def fill_color(frame, box, plane_areas):
    # 提取box的细节
    x, y, w, h = [int(v) for v in box]

    # 确保坐标和宽高不超出图像的范围
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

    # 从plane_areas图中提取对应的区域
    plane_region = plane_areas[y:y+h, x:x+w]

    # 创建一个掩码，其中不是黑色的部分为需要填充的区域
    mask = np.any(plane_region != [0, 0, 0], axis=-1)

    # 使用plane_areas的颜色来覆盖frame中的相应区域
    frame[y:y+h, x:x+w][mask] = plane_region[mask]
    
# 主函数
if __name__ == "__main__":
    video_path = 'testcase1/test9.mp4'
    normal_map = 'testcase1/test9_frame_normal_resize.png'
    poster = 'testcase1/poster.png'
    main(video_path, normal_map, poster)
