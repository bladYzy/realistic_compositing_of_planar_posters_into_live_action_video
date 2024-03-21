import cv2
import numpy as np

# 全局变量
points = []  # 存储选取的点
perspective_ready = False  # 透视变换是否准备好
wall_points = []  # 墙面的四个点


def draw_circle(event, x, y, flags, param):
    global points, perspective_ready, wall_points

    # 右键点击事件
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(points) < 4:  # 先选择墙面的四个角点
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            wall_points.append([x, y])
        elif len(points) == 4:  # 选择矩形的对角顶点
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        else:  # 选择第二个矩形对角顶点并进行透视变换
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            apply_perspective_transform(wall_points, points[4:], img)
            perspective_ready = True

        points.append([x, y])


def apply_perspective_transform(wall_points, rect_points, img):
    # 透视变换
    pts1 = np.float32(wall_points)
    pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])  # 假定墙面映射到的新平面

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (300, 300))

    # 计算变换后的矩形点
    rect_pts = np.float32(rect_points).reshape(-1, 1, 2)
    transformed_rect_pts = cv2.perspectiveTransform(rect_pts, matrix)

    # 在变换后的图像上绘制矩形
    transformed_rect_pts = transformed_rect_pts.reshape(-1, 2).astype(int)
    cv2.rectangle(result, tuple(transformed_rect_pts[0]), tuple(transformed_rect_pts[1]), (255, 0, 0), 3)
    cv2.imshow("Perspective Transform", result)


# 加载图像
img = cv2.imread("sample1.jpg")  # 更改为你的图片路径
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while (True):
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:  # 按下ESC退出
        break

cv2.destroyAllWindows()
