import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, video_path, poster_path='poster.png', zoom_scale=5, zoom_window_size=(400, 400)):
        self.video_path = video_path
        self.poster_path = poster_path
        self.zoom_scale = zoom_scale
        self.zoom_window_size = zoom_window_size
        self.points = []
        self.drawing = False
        self.rect_done = False
        self.mouse_pos = [0, 0]
        self.corners = []
        self.setup_video()

    def setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        ret, self.frame = self.cap.read()
        if not ret:
            print("Failed to load video")
            exit()
        cv2.imshow('Select 4 Points', self.frame)
        cv2.setMouseCallback('Select 4 Points', self.select_points)

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
                # Manually add a point if no corners are being used or detected
                if len(self.points) < 4:
                    self.points.append((x, y))
                    print("Point added manually:", (x, y))
                    if len(self.points) == 4:
                        self.rect_done = True
                        print("Rectangle selection completed.")

    def detect_corners(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert to floating-point for corner detection
        gray = np.float32(gray)
        # Apply the Harris corner detector
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        # Result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in the image with red

        # Coordinates where corners are detected
        corners = np.argwhere(dst > 0.01 * dst.max())
        # Convert coordinates to (x, y) tuples
        corners = [(int(y), int(x)) for x, y in corners]

        return image, corners

    def magnify_region(self, frame, x, y):
        # Calculate the size of the zoom region based on the current zoom scale
        zoom_size = max(100 // self.zoom_scale, 1)
        # Ensure the coordinates for the zoomed region do not go out of frame boundaries
        x1 = int(max(x - zoom_size, 0))
        y1 = int(max(y - zoom_size, 0))
        x2 = int(min(x + zoom_size, frame.shape[1]))
        y2 = int(min(y + zoom_size, frame.shape[0]))

        # Extract the region of interest (ROI) from the frame based on the calculated coordinates
        zoom_region = frame[y1:y2, x1:x2]

        # Resize the extracted region to the size of the zoom window
        zoom_region = cv2.resize(zoom_region, self.zoom_window_size, interpolation=cv2.INTER_LINEAR)

        # Calculate the center position of the magnified area for overlay purposes
        center_x = self.zoom_window_size[0] // 2
        center_y = self.zoom_window_size[1] // 2

        # Optionally, add crosshair for better visual cue of the center
        cv2.line(zoom_region, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)
        cv2.line(zoom_region, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)

        return zoom_region, (x1, y1, center_x, center_y)

    def calculate_rect_size(self, points):
        if len(points) < 4:
            return 0, 0  # Ensure there are exactly four points to form a rectangle

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

    def select_points_interactively(self):
        print("Select 4 Points by clicking in the video window.")
        while len(self.points) < 4:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('i'):  # Increase zoom
                self.zoom_scale += 0.5
            elif key == ord('k'):  # Decrease zoom
                self.zoom_scale = max(1, self.zoom_scale - 0.5)

            if key in [ord('d'), ord('f')]:  # 'd' for regular magnification, 'f' for feature detection
                zoom_region, (x1, y1, center_x, center_y) = self.magnify_region(self.frame, *self.mouse_pos)
                if key == ord('f'):
                    zoom_region, corners = self.detect_corners(zoom_region)
                    self.corners = [(int(c[0] / self.zoom_scale + x1), int(c[1] / self.zoom_scale + y1)) for c in
                                    corners]
                else:
                    self.corners = []

                cv2.imshow('Magnifier', zoom_region)

            cv2.imshow('Select 4 Points', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
                break

    def process_frames(self):
        if len(self.points) < 4:
            print("Insufficient points selected. Exiting processing.")
            return

        print("Processing video frames.")
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                print("No more video frames, or error reading the video.")
                break

            # Optional: Here you might implement tracking or transforming operations
            # For example, applying a perspective transformation or tracking points
            # This is a placeholder for additional processing like:
            # - Update display
            # - Track features
            # - Apply transformations
            # self.apply_transformations()  # Hypothetical method

            cv2.imshow('Video Feed', self.frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cv2.destroyAllWindows()

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

