import cv2
import numpy as np

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(param[0]) < 4:
        param[0].append((x, y))

def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.random.randn(4, 1)
    return kf

def update_kalman_filter(kf, measurement=None):
    kf.predict()
    if measurement is not None:
        measurement = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        kf.correct(measurement)
    predicted_position = prediction[:2].reshape(-1)
    return predicted_position

video_path = 'test6.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Failed to open the video.")
    exit()

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
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        exit()
cv2.destroyAllWindows()

points = np.float32(points)
kfs = [create_kalman_filter() for _ in range(4)]

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)

    for i, (new, status) in enumerate(zip(new_points, status.flatten())):
        if status:
            points[i] = update_kalman_filter(kfs[i], new)
        else:
            points[i] = update_kalman_filter(kfs[i])

    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    frame = cv2.polylines(frame, [np.int32(points).reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow('Tracked', frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
        break

    old_gray = gray_frame.copy()

cap.release()
out.release()
cv2.destroyAllWindows()
