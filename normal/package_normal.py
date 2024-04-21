import cv2
import numpy as np
from get_normal import GetNormal
from PIL import Image

class NormalVideoProcessor:
    def __init__(self, video_path, poster_path, height):
        self.video_path = video_path
        self.poster_path = poster_path
        self.height = height
        self.points = []
        self.drawing = False
        self.rect_done = False
        self.frame = None
        self.cap = None
        self.setup_video()

    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.drawing = True
                self.points.append((x, y))
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Select two top point', self.frame)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
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

    def calculate_average_normal(self, rectangle_points, normal_map):
        normals = [normal_map[int(p[1]), int(p[0])] for p in rectangle_points]
        mean_normal = np.mean(normals, axis=0)
        return self.normalize(mean_normal)

    def calculate_remaining_points(points, height):
        top_left, top_right = points
        bottom_left = (top_left[0], top_left[1] + height)
        bottom_right = (top_right[0], top_right[1] + height)
        return [top_left, top_right, bottom_left, bottom_right]

    def map_points_to_plane(self, rectangle_points, target_normal):
        return [self.project_to_plane(point, target_normal)[:2] for point in rectangle_points]

    def calculate_rect_height(self, points):
        return np.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2)

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def norma_pipline(self):
        normal_generator = GetNormal(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), 'test4')
        normal_map_image = normal_generator.run()
        normal_map = np.array(normal_map_image)
        normal_map = cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)
        return normal_map

    def point_pipline(self):
        self.select_points()



        return self.points

    def setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.ret, self.frame = self.cap.read()

