import cv2
import numpy as np
from get_normal import GetNormal
from PIL import Image

class VideoPointMapper:
    def __init__(self, video_path, normal_map_path, poster_path):
        self.video_path = video_path
        self.normal_map_path = normal_map_path
        self.poster_path = poster_path
        self.points = []
        self.rect_done = False
        self.load_video()

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to load video")
        self.normal_map = self.load_normal_map()
        cv2.imshow('Select two top point', self.frame)
        cv2.setMouseCallback('Select two top point', self.select_points)

    def load_normal_map(self):
        normal_generator = GetNormal(Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)), self.normal_map_path)
        normal_map_image = normal_generator.run()
        normal_map = np.array(normal_map_image)
        return cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)

    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Select two top point', self.frame)
        elif event == cv2.EVENT_LBUTTONUP:
            if len(self.points) == 2:
                self.rect_done = True

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def project_to_plane(self, point, plane_normal):
        point_3d = np.append(point, 0)
        plane_normal = self.normalize(plane_normal)
        distance = np.dot(point_3d, plane_normal)
        return point_3d - distance * plane_normal

    def calculate_remaining_points(self, height):
        top_left, top_right = self.points
        bottom_left = (top_left[0], top_left[1] + height)
        bottom_right = (top_right[0], top_right[1] + height)
        return [top_left, top_right, bottom_right, bottom_left]

    def calculate_average_normal(self, rectangle_points):
        normals = [self.normal_map[int(p[1]), int(p[0])] for p in rectangle_points]
        mean_normal = np.mean(normals, axis=0)
        return self.normalize(mean_normal)

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calculate_rect_size(self, points):
        width = (self.calculate_distance(points[0], points[1]) + self.calculate_distance(points[2], points[3])) / 2
        height = (self.calculate_distance(points[0], points[3]) + self.calculate_distance(points[1], points[2])) / 2
        return int(width), int(height)

    def run(self):
        while not self.rect_done:
            cv2.waitKey(1)
        cv2.destroyAllWindows()

        # Get rectangle height from user input
        height = int(input("Enter the rectangle height: "))
        all_points = self.calculate_remaining_points(height)
        normal = self.calculate_average_normal(all_points)
        mapped_points = [self.project_to_plane(point, normal)[:2] for point in all_points]

        print("All Points:", mapped_points)
        points = np.array(mapped_points, dtype=np.float32).reshape(-1, 1, 2)

        rect_size = self.calculate_rect_size(all_points)

        # Load and resize the poster image
        poster = cv2.imread(self.poster_path)
        poster = cv2.resize(poster, rect_size)

        # Setup for optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        old_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Set up video writer
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output_path = 'output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Points for perspective transformation
        poster_points = np.float32([[0, 0], [rect_size[0], 0], [rect_size[0], rect_size[1]], [0, rect_size[1]]])

        # Video processing loop
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)
            good_points = points[status.flatten() == 1]

            if len(good_points) < 4:
                print("Not enough points tracked!")
                break

            # Perspective transformation
            M = cv2.getPerspectiveTransform(poster_points, good_points.reshape(-1, 2))
            warped_poster = cv2.warpPerspective(poster, M, (frame.shape[1], frame.shape[0]))

            # Create a mask where the poster will be placed
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(good_points), (255,) * frame.shape[2])
            inv_mask = cv2.bitwise_not(mask)

            # Combine the current frame and the warped poster
            frame = cv2.bitwise_and(frame, inv_mask)
            frame = cv2.bitwise_or(frame, warped_poster)

            cv2.imshow('Tracked', frame)
            out.write(frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
                break

            old_gray = gray_frame.copy()
            points = good_points.reshape(-1, 1, 2)

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = 'test4/test4.mp4'
    normal_map_path = 'test4'
    poster_path = 'test3/poster.png'
    mapper = VideoPointMapper(video_path, normal_map_path, poster_path)
    mapper.run()
