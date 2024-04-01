import cv2
import numpy as np

def select_points(event, x, y, flags, param):
    # Accessing the frame and points list from the passed dictionary
    frame = param['frame']
    points = param['points']
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw the selected point
        if len(points) > 1:
            cv2.line(frame, points[-1], points[-2], (255, 0, 0), 2)  # Connect the last two points
        cv2.imshow('Select 4 Points', frame)  # Update the frame display

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_rect_size(points):
    width = (calculate_distance(points[0], points[1]) + calculate_distance(points[2], points[3])) / 2
    height = (calculate_distance(points[0], points[3]) + calculate_distance(points[1], points[2])) / 2
    return int(width), int(height)

video_path = 'test9.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

output_path = 'test_8.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Display the first frame and wait for the user to select 4 points
drawing_frame = frame.copy()  # Copy the frame for drawing
points = []
param = {'frame': drawing_frame, 'points': points}  # Dictionary for parameters
cv2.imshow('Select 4 Points', drawing_frame)
cv2.setMouseCallback('Select 4 Points', select_points, param)

while len(points) < 4:
    cv2.waitKey(1)
cv2.destroyAllWindows()

points = np.float32(points)
rect_size = calculate_rect_size(points)

poster = cv2.imread('poster.png')
poster = cv2.resize(poster, rect_size)

poster_points = np.float32([[0, 0], [rect_size[0], 0], [rect_size[0], rect_size[1]], [0, rect_size[1]]])

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)
    good_points = points[status.flatten() == 1]

    M = cv2.getPerspectiveTransform(poster_points, good_points)
    warped_poster = cv2.warpPerspective(poster, M, (frame.shape[1], frame.shape[0]))

    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(good_points), (255,) * frame.shape[2])
    inv_mask = cv2.bitwise_not(mask)
    frame = cv2.bitwise_and(frame, inv_mask)
    frame = cv2.bitwise_or(frame, warped_poster)

    cv2.imshow('Tracked', frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    old_gray = gray_frame.copy()
    points = good_points.reshape(-1, 1, 2)

cap.release()
out.release()
cv2.destroyAllWindows()
