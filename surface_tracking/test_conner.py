import cv2
import numpy as np

point_selected = False
points = []


def select_point(event, x, y, flags, params):
    global point_selected, points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        point_selected = True


cap = cv2.VideoCapture('test6.mp4')
ret, first_frame = cap.read()

if not cap.isOpened():
    print("Error opening video stream or file")
cv2.imshow('First Frame - Select 4 Points and Press ENTER', first_frame)
cv2.setMouseCallback('First Frame - Select 4 Points and Press ENTER', select_point)

while True:
    cv2.imshow('First Frame - Select 4 Points and Press ENTER', first_frame)
    if len(points) == 4:
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) & 0xFF == 13:  # 13 is the Enter Key
        break

#to np
points = np.array(points, dtype=np.float32)


old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 光流追踪
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# kashi追踪
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算新的点位置
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)
    old_gray = gray_frame.copy()
    points = new_points

    # 画点er
    for i, (new) in enumerate(new_points):
        x, y = new.ravel()
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == 27:  # 27 is the Esc Key
        break

cap.release()
cv2.destroyAllWindows()
