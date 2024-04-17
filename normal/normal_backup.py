import cv2
import numpy as np

# Initialize global variables
points = []
drawing = False  # True if mouse is pressed
rect_done = False  # True if rectangle selection is complete

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype="float32")


# Mouse callback function to select points in the video
def select_points(event, x, y, flags, param):
    global points, drawing, rect_done, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:  # 限制点的选择只到两个
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 显示选定的点
            if len(points) == 2:
                rect_done = True  # 两个点选完后完成矩形选择
            cv2.imshow('Select Area', frame)

# Calculate the other two points based on height
def calculate_other_two_points(p1, p2, height):
    vector = np.array(p2) - np.array(p1)
    perpendicular_vector = np.array([-vector[1], vector[0]])
    unit_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
    p3 = p1 + unit_vector * height
    p4 = p2 + unit_vector * height
    return np.int32(p3), np.int32(p4)

# Calculate average normal in a selected region
def calculate_average_normal(normals, points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    region = normals[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    avg_normal = np.mean(region, axis=(0, 1))
    return avg_normal

# Load video
video_path = 'test3/test3.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cv2.imshow('Select Area', frame)
cv2.setMouseCallback('Select Area', select_points)

# Wait until the rectangle is selected
while not rect_done:
    cv2.waitKey(1)
cv2.destroyAllWindows()

# Input the height of the rectangle from the user
height = int(input("Please enter the height of the rectangle: "))

# Calculate the remaining two points
p3, p4 = calculate_other_two_points(np.array(points[0]), np.array(points[1]), height)
points.extend([p3, p4])  # Add these points to the points list

# Set up video writer
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Read and resize the poster to the size of the selected rectangle
poster_path = 'test3/poster.png'
poster = cv2.imread(poster_path)
rect_width = abs(points[0][0] - points[1][0])
rect_height = abs(points[0][1] - points[1][1])
poster = cv2.resize(poster, (rect_width, rect_height))

# Poster points and perspective transformation setup
poster_points = np.float32([[0, 0], [rect_width, 0], [rect_width, rect_height], [0, rect_height]])

# Initialize optical flow parameters
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
points = np.float32(points).reshape(-1, 1, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None)
    good_points = points[status.flatten() == 1]

    if len(good_points) >= 4:  # 确保有足够的点进行透视变换
        ordered_points = order_points(good_points.reshape(4, 2))
        M = cv2.getPerspectiveTransform(poster_points, ordered_points)
        warped_poster = cv2.warpPerspective(poster, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)

        # Composite the transformed poster onto the frame
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(good_points.reshape(-1, 2)), (255,) * frame.shape[2])
        inv_mask = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, inv_mask)
        frame = cv2.bitwise_or(frame, warped_poster)
    else:
        print("Not enough points tracked for perspective transformation.")

    cv2.imshow('Tracked', frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    old_gray = gray_frame.copy()
    points = good_points.reshape(-1, 1, 2)

cap.release()
out.release()
cv2.destroyAllWindows()
