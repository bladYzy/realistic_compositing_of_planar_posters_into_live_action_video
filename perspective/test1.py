import cv2
import numpy as np

# 全局变量
points = []  # 存储用户点击的点

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # 红色表示选择的点
        points.append((x, y))
        if len(points) == 4:  # 选择了四个点后自动绘制中心矩形
            draw_center_rectangle()
        cv2.imshow("Image", img)

def draw_center_rectangle():
    global points, img
    # 计算中心点
    center_point = np.mean(points, axis=0).astype(int)
    width = 50
    height = 30
    top_left = (center_point[0] - width // 2, center_point[1] - height // 2)
    bottom_right = (center_point[0] + width // 2, center_point[1] + height // 2)
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)  # 蓝色表示绘制的矩形
    # 透视变换
    perform_perspective_transform()

def perform_perspective_transform():
    global points, img
    # 目标矩形的四个角点
    width, height = 200, 100
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(np.float32(points), dst_points)
    # 应用透视变换
    warped = cv2.warpPerspective(img, M, (width, height))
    cv2.imshow("Warped Image", warped)

# 主程序
if __name__ == "__main__":
    # 加载图片
    img = np.zeros((512, 512, 3), np.uint8)  # 创建一个黑色的图像
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出
            break

    cv2.destroyAllWindows()
