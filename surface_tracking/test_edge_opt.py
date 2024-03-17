import cv2
import numpy as np


def select_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(params[0]) < 4:
        params[0].append((x, y))


def estimate_missing_point(pts):
    if len(pts) != 3:
        return None
    p1, p2, p3 = pts[0], pts[1], pts[2]
    p4 = p1 + p3 - p2
    return p4


video_path = 'test6.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()


fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


output_path = 'tracked_video_6.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)


cv2.imshow('Select 4 Points', frame)
points = []
cv2.setMouseCallback('Select 4 Points', select_points, [points])

while len(points) < 4:
    cv2.imshow('Select 4 Points', frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()

points = np.float32(points)
initial_points = points.copy()

# 追
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)

    good_points = []
    for i, (new, good) in enumerate(zip(new_points, status.flatten())):
        if good:
            good_points.append(new)
        else:
            # 估算缺失点
            remaining_points = [good_points[i] for i in range(len(good_points)) if i != i]
            estimated = estimate_missing_point(remaining_points)
            if estimated is not None:
                good_points.append(estimated)
            else:
                good_points.append(points[i])  # 用上一个位置代替

    good_points = np.array(good_points, dtype=np.float32)

    # 绘制
    for i, point in enumerate(good_points):
        x, y = point.ravel()
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    pts = good_points.reshape((-1, 1, 2))
    frame = cv2.polylines(frame, [np.int32(pts)], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow('Tracked', frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    old_gray = gray_frame.copy()
    points = good_points

cap.release()
out.release()  # 关闭VideoWriter对象
cv2.destroyAllWindows()
