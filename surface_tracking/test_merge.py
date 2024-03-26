import cv2
import numpy as np


def select_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(params[0]) < 4:
        params[0].append((x, y))


def calculate_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    return normal / norm


video_path = 'test7.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

output_path = 'test_7.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

cv2.imshow('Select 4 Points', frame)
points = []
cv2.setMouseCallback('Select 4 Points', select_points, [points])

while len(points) < 4:
    cv2.waitKey(1)
cv2.destroyAllWindows()

points = np.float32(points)
initial_points = points.copy()

poster = cv2.imread('poster.jpg')
poster_size = (frame.shape[1], frame.shape[0])
poster_points = np.float32([[0, 0], [poster_size[0], 0], [poster_size[0], poster_size[1]], [0, poster_size[1]]])

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)

    good_points = new_points[status.flatten() == 1]

    # 计算透视
    M = cv2.getPerspectiveTransform(poster_points, good_points)
    warped_poster = cv2.warpPerspective(poster, M, (frame.shape[1], frame.shape[0]))

    # 合并
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(good_points), (255,) * frame.shape[2])
    inv_mask = cv2.bitwise_not(mask)
    frame = cv2.bitwise_and(frame, inv_mask)
    frame = cv2.bitwise_or(frame, warped_poster)

    cv2.imshow('Tracked', frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    old_gray = gray_frame.copy()
    points = good_points

cap.release()
out.release()
cv2.destroyAllWindows()
