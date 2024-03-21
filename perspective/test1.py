import cv2
import numpy as np

# 全局变量
refPt = []  # 存储透视变换的参考点
scaling_factor = 1.0  # 图像缩放因子
dragging = False  # 是否正在拖动图像
orig_img = None  # 原始图像
img = None  # 当前显示的图像（可能被移动或缩放）
dx, dy = 0, 0  # 图像移动的偏移量

def click_and_crop(event, x, y, flags, param):
    global refPt, dragging, dx, dy, scaling_factor, img

    if event == cv2.EVENT_RBUTTONDOWN:
        if len(refPt) < 4:
            # 选择透视变换的四个点
            refPt.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        if len(refPt) == 4:
            # 应用透视变换
            apply_perspective_transform()
    elif event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        dx, dy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        img = np.roll(orig_img, (int((y-dy)*scaling_factor), int((x-dx)*scaling_factor)), axis=(0,1))
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scaling_factor *= 1.1
        else:
            scaling_factor /= 1.1
        img = cv2.resize(orig_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

def apply_perspective_transform():
    global refPt, orig_img, img, scaling_factor
    # 定义透视变换的目标点
    pts_dst = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(np.float32(refPt), pts_dst)
    img = cv2.warpPerspective(orig_img, M, (img.shape[1], img.shape[0]))
    orig_img = img.copy()
    scaling_factor = 1.0
    refPt = []

def reset_image():
    global orig_img, img, scaling_factor, refPt
    img = orig_img.copy()
    scaling_factor = 1.0
    refPt = []

# 加载图像
orig_img = cv2.imread("sample1.jpg")
img = orig_img.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# 主循环
while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF

    # 按 'esc' 重置图像
    if key == 27:
        reset_image()

cv2.destroyAllWindows()
