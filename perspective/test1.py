import cv2
import numpy as np

# 全局变量
points = []  # 存储选定的点
src_pts = []  # 墙面四个角点
dst_pts = []  # 矩形的两个对角点
img = None
original_img = None
perspective_transform_matrix = None
drag_start = None
zoom_level = 1.0
img_position = [0, 0]

# 鼠标事件回调函数
def mouse_event(event, x, y, flags, param):
    global src_pts, dst_pts, img, original_img, perspective_transform_matrix, drag_start, zoom_level, img_position

    if event == cv2.EVENT_RBUTTONDOWN:
        if len(src_pts) < 4:
            # 选择墙面的四个角点
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            src_pts.append([x, y])
            if len(src_pts) == 4:
                # 计算透视变换矩阵
                dst = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], np.float32)
                perspective_transform_matrix = cv2.getPerspectiveTransform(np.float32(src_pts), dst)
        elif len(dst_pts) < 2:
            # 选择矩形的对角点并进行透视变换
            dst_pts.append([x, y])
            if len(dst_pts) == 2:
                # 应用透视变换到选择的点
                pts = np.float32([dst_pts]).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(pts, perspective_transform_matrix)
                # 在原图上绘制变换后的矩形
                transformed_pts = transformed_pts.reshape(-1, 2)
                cv2.polylines(img, [np.int32(transformed_pts)], True, (255, 0, 0), 3)
                dst_pts.clear()  # 清除点以重新开始

    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_start:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            img_position[0] += dx
            img_position[1] += dy
            drag_start = [x, y]
            update_img()

    elif event == cv2.EVENT_LBUTTONDOWN:
        drag_start = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        drag_start = None

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom_level *= 1.1
        else:
            zoom_level /= 1.1
        update_img()

# 更新显示图像
def update_img():
    global img, original_img, zoom_level, img_position
    nh, nw = int(original_img.shape[0] * zoom_level), int(original_img.shape[1] * zoom_level)
    resized_img = cv2.resize(original_img, (nw, nh))
    x, y = img_position
    h, w = img.shape[:2]
    x = min(max(0, x), nw - w)
    y = min(max(0, y), nh - h)
    img_position = [x, y]
    img = resized_img[y:y+h, x:x+w]

def main(image_path):
    global img, original_img
    original_img = cv2.imread(image_path)
    img = original_img.copy()
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Image', mouse_event)

    while True:
        cv2.imshow('Image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('path_to
