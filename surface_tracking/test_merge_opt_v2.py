import cv2
import numpy as np

#to 选择四个点
# Define mouse event callback function to select points in the video
# 更新鼠标回调函数以记录鼠标位置
def select_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(params[0]) < 4:
        params[0].append((x, y))
    params[1][0] = x
    params[1][1] = y

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype="float32")

#两点距离
# Calculate distance
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

#计算矩形的尺寸
#size of rect
def calculate_rect_size(points):
    width = (calculate_distance(points[0], points[1]) + calculate_distance(points[2], points[3])) / 2
    height = (calculate_distance(points[0], points[3]) + calculate_distance(points[1], points[2])) / 2
    return int(width), int(height)

video_path = 'test7.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Get the FPS and size of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

output_path = 'test_7.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

#Display the first frame of the video
cv2.imshow('Select 4 Points', frame)
points = []
mouse_pos = [0, 0]
cv2.setMouseCallback('Select 4 Points', select_points, [points,mouse_pos])

while len(points) < 4:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        # 计算放大区域
        x, y = mouse_pos
        zoom_scale = 2  # 放大倍数
        zoom_size = 100  # 放大区域的半径
        x1, y1 = max(x - zoom_size, 0), max(y - zoom_size, 0)
        x2, y2 = min(x + zoom_size, frame.shape[1]), min(y + zoom_size, frame.shape[0])
        zoom_region = frame[y1:y2, x1:x2]
        zoom_region = cv2.resize(zoom_region, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LINEAR)

        # 在放大区域的中心添加一个标记
        center_x, center_y = zoom_region.shape[1] // 2, zoom_region.shape[0] // 2
        cv2.drawMarker(zoom_region, (center_x, center_y), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10,
                       thickness=1)

        # 显示放大镜视图
        cv2.imshow('Magnifier', zoom_region)

cv2.destroyAllWindows()

points = np.float32(points)
rect_size = calculate_rect_size(points)

#load and resize poster
poster = cv2.imread('poster.png')
poster = cv2.resize(poster, rect_size)

poster_points = np.float32([[0, 0], [rect_size[0], 0], [rect_size[0], rect_size[1]], [0, rect_size[1]]])

# Set parameters for LK
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)
    good_points = points[status.flatten() == 1]
    #ordered_points = order_points(good_points.reshape(4, 2))
    M = cv2.getPerspectiveTransform(poster_points, good_points)
    warped_poster = cv2.warpPerspective(poster, M, (frame.shape[1], frame.shape[0]))

    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(good_points), (255,) * frame.shape[2])
    inv_mask = cv2.bitwise_not(mask)
    frame = cv2.bitwise_and(frame, inv_mask)
    frame = cv2.bitwise_or(frame, warped_poster)

    cv2.imshow('Tracked', frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    old_gray = gray_frame.copy()
    points = good_points.reshape(-1, 1, 2)

cap.release()
out.release()
cv2.destroyAllWindows()