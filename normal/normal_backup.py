import cv2
import numpy as np
from get_normal import GetNormal
from PIL import Image

# Initialize global variables
points = []
drawing = False  # True if mouse is pressed
rect_done = False  # True if rectangle selection is complete


# Mouse callback function to select points in the video
def select_points(event, x, y, flags, param):
    global points, drawing, rect_done, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:  # Ensure only two points are selected
            drawing = True
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select two top point', frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(points) == 2:
            rect_done = True

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def project_to_plane(point, plane_normal):
    point_3d = np.append(point, 0)
    plane_normal = normalize(plane_normal)
    distance = np.dot(point_3d, plane_normal)
    return point_3d - distance * plane_normal



def calculate_average_normal(rectangle_points, normal_map):
    normals = [normal_map[int(p[1]), int(p[0])] for p in rectangle_points]
    mean_normal = np.mean(normals, axis=0)
    return normalize(mean_normal)


def calculate_remaining_points(points, height):

    top_left, top_right = points
    bottom_left = (top_left[0], top_left[1] + height)
    bottom_right = (top_right[0], top_right[1] + height)
    return [top_left, top_right, bottom_right, bottom_left]



def map_points_to_plane(rectangle_points, target_normal):
    return [project_to_plane(point, target_normal)[:2] for point in rectangle_points]


#计算矩形的尺寸
#size of rect
def calculate_rect_size(points):
    width = (calculate_distance(points[0], points[1]) + calculate_distance(points[2], points[3])) / 2
    height = (calculate_distance(points[0], points[3]) + calculate_distance(points[1], points[2])) / 2
    return int(width), int(height)

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Load video and normal map
video_path = 'test4/test4.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()


normal_generator = GetNormal(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), 'test4')
normal_map_image = normal_generator.run()
normal_map = np.array(normal_map_image)
normal_map = cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)

# Setup display window and mouse callback
cv2.imshow('Select two top point', frame)
cv2.setMouseCallback('Select two top point', select_points)

# Wait until the rectangle is selected
while not rect_done:
    cv2.waitKey(1)
cv2.destroyAllWindows()

# Ask user for rectangle height and calculate remaining points
while True:
    try:
        height = int(input("Enter the rectangle height: "))
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

# 确保在调用光流算法之前将点转换为正确的格式
all_points = calculate_remaining_points(points, height)
normal = calculate_average_normal(all_points, normal_map)
all_points = map_points_to_plane(all_points, normal)

print("All Points:", all_points)

# 将点转换为适当的数据类型和形状
points = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)

rect_size = calculate_rect_size(all_points)

# Set up video writer
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Read and resize the poster to the size of the selected rectangle
poster_path = 'test3/poster.png'
poster = cv2.imread(poster_path)
poster = cv2.resize(poster, rect_size)

# Poster points and perspective transformation setup
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
cv2.destroyAllWindows()