import cv2
import numpy as np

drawing = False  # 如果按下鼠标，则为真
ix, iy = -1, -1
perspective_points = []
original_image = cv2.imread("sample1.jpg")
img = original_image.copy()
zoom_level = 1.0
moved = False


# 鼠标回调函数
def draw_rectangle_with_perspective(event, x, y, flags, param):
    global ix, iy, drawing, perspective_points, img, zoom_level, moved

    if event == cv2.EVENT_RBUTTONDOWN:
        if len(perspective_points) < 4:
            # 将鼠标位置调整到缩放和移动后的坐标
            adjusted_x, adjusted_y = int(x / zoom_level), int(y / zoom_level)
            perspective_points.append((adjusted_x, adjusted_y))
            cv2.circle(img, (adjusted_x, adjusted_y), 5, (0, 0, 255), -1)
        else:
            print("All perspective points are selected. Press 'r' to reset.")

    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_img = img.copy()
        cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
        show_image(temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)


def show_image(image, title='image'):
    global zoom_level
    resized_image = cv2.resize(image, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(title, resized_image)


def reset_image():
    global img, perspective_points, zoom_level
    img = original_image.copy()
    perspective_points = []
    zoom_level = 1.0
    show_image(img)


def apply_perspective_transform(image, points):
    if len(points) != 4:
        return image
    # 源四边形坐标
    pts1 = np.float32(points)
    # 目标四边形坐标
    pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    # 获取透视变换矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 应用透视变换
    result = cv2.warpPerspective(image, matrix, (300, 300))
    return result


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle_with_perspective)

while True:
    show_image(img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # 按Esc键退出
        break
    elif k == ord('r'):  # 重置图像
        reset_image()
    elif k == ord('+') and zoom_level < 3.0:  # 放大图像
        zoom_level += 0.1
    elif k == ord('-') and zoom_level > 0.5:  # 缩小图像
        zoom_level -= 0.1
    elif k == ord('p') and len(perspective_points) == 4:  # 应用透视变换
        warped_image = apply_perspective_transform(original_image, perspective_points)
        cv2.imshow("Perspective Transform", warped_image)

cv2.destroyAllWindows()
