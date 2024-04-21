import cv2
import numpy as np

class PerspectiveVideoProcessor:
    def __init__(self, video_path, poster_path, zoom_scale=5, zoom_window_size=(400, 400)):
        self.video_path = video_path
        self.poster_path = poster_path
        self.zoom_scale = zoom_scale
        self.zoom_window_size = zoom_window_size
        self.points = []
        self.drawing = False
        self.rect_done = False
        self.mouse_pos = [0, 0]
        self.corners = []
        self.poster = cv2.imread(self.poster_path)

        if self.poster is None:
            print("Failed to load poster image.")
            exit()
        self.setup_video()

    def setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        ret, self.frame = self.cap.read()
        if not ret:
            print("Failed to load video")
            exit()
        self.old_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Select 4 Points', self.frame)
        cv2.setMouseCallback('Select 4 Points', self.select_points)

    def detect_corners(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        corners = np.argwhere(dst > 0.01 * dst.max())
        corners = [(int(y), int(x)) for x, y in corners]
        image[dst > 0.01 * dst.max()] = [0, 0, 255]

        # 在检测角点后绘制十字箭头
        center_x = self.zoom_window_size[0] // 2
        center_y = self.zoom_window_size[1] // 2
        cv2.line(image, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
        cv2.line(image, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)

        return image, corners

    def magnify_region(self, frame, x, y):
        zoom_size = max(100 // self.zoom_scale, 1)
        x1 = max(x - zoom_size, 0)
        y1 = max(y - zoom_size, 0)
        x2 = min(x + zoom_size, frame.shape[1])
        y2 = min(y + zoom_size, frame.shape[0])

        zoom_region = frame[y1:y2, x1:x2]
        zoom_region_resized = cv2.resize(zoom_region, self.zoom_window_size, interpolation=cv2.INTER_LINEAR)

        return zoom_region_resized, (x1, y1, x2, y2)

    def calculate_rect_size(self, points):
        if len(points) < 4:
            return 0, 0

        # Calculate width as the average of distances between points [0] -> [1] and [2] -> [3]
        width = (self.calculate_distance(points[0], points[1]) +
                 self.calculate_distance(points[2], points[3])) / 2

        # Calculate height as the average of distances between points [0] -> [3] and [1] -> [2]
        height = (self.calculate_distance(points[0], points[3]) +
                  self.calculate_distance(points[1], points[2])) / 2

        # Return the average width and height as integer values
        return int(width), int(height)

    def calculate_distance(self, p1, p2):
        # Calculate Euclidean distance between two points (p1 and p2)
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def select_points(self, event, x, y, flags, params):
        # Update mouse position
        self.mouse_pos[0] = x
        self.mouse_pos[1] = y

        # Handle mouse events for point selection
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.corners:
                # Select the nearest corner if corners are detected
                min_distance = float('inf')
                nearest_corner = None
                for corner in self.corners:
                    cx, cy = corner
                    distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_corner = (int(cx), int(cy))
                if nearest_corner and min_distance < 20:  # Assume a threshold of 20 pixels for selection
                    self.points.append(nearest_corner)
                    print("Corner detected and added:", nearest_corner)
                    if len(self.points) == 4:
                        self.rect_done = True
                        print("Rectangle selection completed.")
            else:
                if len(self.points) < 4:
                    self.points.append((x, y))
                    print("Point added manually:", (x, y))
                    if len(self.points) == 4:
                        self.rect_done = True
                        print("Rectangle selection completed.")

    def select_points_interactively(self):
        print("Select 4 Points by clicking in the video window.")
        while len(self.points) < 4:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('i'):  # Increase zoom
                self.zoom_scale += 0.5
            elif key == ord('k'):  # Decrease zoom
                self.zoom_scale = max(1, self.zoom_scale - 0.5)

            if key == ord('o'):  # Regular magnification with crosshairs
                zoom_region, _ = self.magnify_region(self.frame, *self.mouse_pos)
                center_x = self.zoom_window_size[0] // 2
                center_y = self.zoom_window_size[1] // 2
                cv2.line(zoom_region, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
                cv2.line(zoom_region, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)
                cv2.imshow('Magnifier', zoom_region)

            elif key == ord('l'):  # Feature detection without initial crosshairs
                zoom_region, _ = self.magnify_region(self.frame, *self.mouse_pos)
                zoom_region, corners = self.detect_corners(zoom_region)
                center_x = self.zoom_window_size[0] // 2
                center_y = self.zoom_window_size[1] // 2
                cv2.line(zoom_region, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
                cv2.line(zoom_region, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)
                cv2.imshow('Magnifier', zoom_region)


            cv2.imshow('Select 4 Points', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def process_frames(self):
        if not hasattr(self, 'old_gray'):
            print("Old gray frame not initialized.")
            return
        if len(self.points) < 4:
            print("Insufficient points selected. Exiting processing.")
            return

        # Define poster_points based on the poster size and intended transformation
        poster_points = np.array([[0, 0], [self.poster.shape[1], 0],
                                  [self.poster.shape[1], self.poster.shape[0]], [0, self.poster.shape[0]]], dtype=np.float32)

        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        print("Processing video frames.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            points, status, error = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, np.float32(self.points), None, **lk_params)
            good_points = points[status.flatten() == 1]
            if len(good_points) == 4:
                M = cv2.getPerspectiveTransform(poster_points, good_points)
                warped_poster = cv2.warpPerspective(self.poster, M, (frame.shape[1], frame.shape[0]))

                mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32(good_points), (255,) * frame.shape[2])
                inv_mask = cv2.bitwise_not(mask)
                frame = cv2.bitwise_and(frame, inv_mask)
                frame = cv2.bitwise_or(frame, warped_poster)

                cv2.imshow('Tracked', frame)

                self.old_gray = gray_frame.copy()
                self.points = good_points.reshape(-1, 1, 2)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    def run(self):
        self.select_points_interactively()
        self.process_frames()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_processor = VideoProcessor('test7.mp4')
    video_processor.run()
    video_processor.cleanup()

