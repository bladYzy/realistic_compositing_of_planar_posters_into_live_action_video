import cv2
import numpy as np

# 初始化点选择的回调函数
point_selected = False
points = []


def select_point(event, x, y, flags, params):
    global point_selected, points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        point_selected = True
        if len(points) == 4:
            cv2.destroyAllWindows()



cap = cv2.VideoCapture('test7.mp4')
ret, first_frame = cap.read()


cv2.imshow('First Frame - Select 4 Points', first_frame)
cv2.setMouseCallback('First Frame - Select 4 Points for the Rectangle', select_point)

while True:
    key = cv2.waitKey(1)
    for i, point in enumerate(points):
        cv2.circle(first_frame, point, 5, (0, 255, 0), -1)
    cv2.imshow('First Frame - Select 4 Points for the Rectangle and Press ENTER', first_frame)
    if len(points) >= 4 or key == 13:
        break

cv2.destroyAllWindows()


points = np.array(points, dtype=np.float32)


old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)
    old_gray = gray_frame.copy()
    points = new_points

    for i, (new) in enumerate(new_points):
        x, y = new.ravel()
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    pts = new_points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == 27:  # 27 is the Esc Key
        break

cap.release()
cv2.destroyAllWindows()
