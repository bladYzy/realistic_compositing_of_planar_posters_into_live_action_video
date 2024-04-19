import cv2
import numpy as np

# Initialize global variables
points = []
drawing = False  # True if mouse is pressed
rect_done = False  # True if rectangle selection is complete
mouse_pos = [0, 0]
corners = []

# Mouse callback function to select points in the video
def select_points(event, x, y, flags, params):
    global points, drawing, rect_done, frame, corners
    mouse_pos[0] = x
    mouse_pos[1] = y

    if event == cv2.EVENT_LBUTTONDOWN:
        if corners:
            # Select the nearest corner
            min_distance = float('inf')
            nearest_corner = None
            for cx, cy in corners:
                distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_corner = (int(cx), int(cy))
            if nearest_corner and min_distance < 20:  # threshold for maximum response distance
                points.append(nearest_corner)
                print("Corner detected and added")
                if len(points) == 4:
                    rect_done = True
        else:
            if len(points) < 4:
                points.append((x, y))
                print("Point added manually")
                if len(points) == 4:
                    rect_done = True

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    return image, np.argwhere(dst > 0.01 * dst.max())

def magnify_region(frame, x, y, zoom_scale, zoom_window_size):
    zoom_size = max(100 // zoom_scale, 1)
    x1 = int(max(x - zoom_size, 0))
    y1 = int(max(y - zoom_size, 0))
    x2 = int(min(x + zoom_size, frame.shape[1]))
    y2 = int(min(y + zoom_size, frame.shape[0]))
    zoom_region = frame[y1:y2, x1:x2]
    zoom_region = cv2.resize(zoom_region, zoom_window_size, interpolation=cv2.INTER_LINEAR)

    center_x = zoom_window_size[0] // 2
    center_y = zoom_window_size[1] // 2

    return zoom_region, x1, y1, center_x, center_y


def calculate_rect_size(points):
    width = (calculate_distance(points[0], points[1]) + calculate_distance(points[2], points[3])) / 2
    height = (calculate_distance(points[0], points[3]) + calculate_distance(points[1], points[2])) / 2
    return int(width), int(height)
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
# Load video
video_path = 'test7.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Failed to load video")
    exit()

# Setup display window and mouse callback
cv2.imshow('Select 4 Points', frame)
cv2.setMouseCallback('Select 4 Points', select_points)

# Default magnifier parameters
zoom_scale = 5
zoom_window_size = (400, 400)

while len(points) < 4:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('i'):  # Increase zoom
        zoom_scale += 0.5
    elif key == ord('k'):  # Decrease zoom
        zoom_scale = max(1, zoom_scale - 0.5)
    # 处理键盘输入和放大区域的更新
    if key in [ord('d'), ord('f')]:  # 'd' for regular magnification, 'f' for feature detection
        zoom_region, x1, y1, center_x, center_y = magnify_region(frame, *mouse_pos, zoom_scale, zoom_window_size)

        if key == ord('f'):
            zoom_region, temp_corners = detect_corners(zoom_region)
            corners = [(c[1] / zoom_scale + x1, c[0] / zoom_scale + y1) for c in temp_corners]
        else:
            corners = []
        cv2.line(zoom_region, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
        cv2.line(zoom_region, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)

        cv2.imshow('Magnifier', zoom_region)

cv2.destroyAllWindows()

# Convert points to float32 for further processing
points = np.float32(points)
# Calculate the rectangle size based on selected points
rect_size = calculate_rect_size(points)

# Resize poster
poster = cv2.imread('poster.png')
poster = cv2.resize(poster, rect_size)

# Poster points for perspective transformation setup
poster_points = np.float32([[0, 0], [rect_size[0], 0], [rect_size[0], rect_size[1]], [0, rect_size[1]]])

# Set parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Start video processing for tracking and applying perspective transformation
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)
    good_points = points[status.flatten() == 1]
    if len(good_points) == 4:
        M = cv2.getPerspectiveTransform(poster_points, good_points)
        warped_poster = cv2.warpPerspective(poster, M, (frame.shape[1], frame.shape[0]))

        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(good_points), (255,) * frame.shape[2])
        inv_mask = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, inv_mask)
        frame = cv2.bitwise_or(frame, warped_poster)

        cv2.imshow('Tracked', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    old_gray = gray_frame.copy()
    points = good_points.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
