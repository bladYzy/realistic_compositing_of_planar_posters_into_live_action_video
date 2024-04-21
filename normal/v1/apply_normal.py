import cv2
import numpy as np


def apply_perspective_transform(image, plane_normal):
    angle = np.arctan2(plane_normal[1], plane_normal[0]) * 180 / np.pi

    h, w = image.shape[:2]

    src_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_points = np.float32([
        [w * 0.05, h * 0.33],
        [w * 0.9, h * 0.25],
        [w * 0.85 + w * 0.1 * np.cos(angle), h * 0.75 + h * 0.1 * np.sin(angle)],
        [w * 0.15, h * 0.85]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(image, matrix, (w, h))

    return transformed_image

image = cv2.imread('path_to_your_image.jpg')
)
plane_normal = np.array([1, 1, 0]) 

# 应用变换
transformed_image = apply_perspective_transform(image, plane_normal)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
