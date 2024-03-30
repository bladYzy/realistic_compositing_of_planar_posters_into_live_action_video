import numpy as np
import cv2
from skimage.filters import sobel_h, sobel_v
import matplotlib.pyplot as plt


def select_roi(image):

    r = cv2.selectROI("Select ROI", image)
    cv2.destroyWindow("Select ROI")
    return r  


def calculate_normal(depth_image, roi):

    x, y, w, h = roi

    depth_roi = depth_image[y:y + h, x:x + w]

    grad_x = sobel_h(depth_roi)
    grad_y = sobel_v(depth_roi)

    normals = np.dstack((-grad_x, -grad_y, np.ones_like(grad_x)))

    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals_normalized = normals / norm

    normal_average = np.mean(normals_normalized, axis=(0, 1))
    return normal_average


def visualize_result(image, roi, normal_vector):

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    # 绘制ROI
    x, y, w, h = roi
    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    center_x, center_y = x + w / 2, y + h / 2

    scale_factor = 200

    # draw
    ax.quiver(center_x, center_y, normal_vector[0], normal_vector[1], color='white', scale=scale_factor, width=0.005)

    print(f"Normal Vector: {normal_vector}")
    print(f"Arrow Position: ({center_x}, {center_y})")
    print(f"Scale Factor: {scale_factor}")

    plt.show()


if __name__ == "__main__":

    depth_image = cv2.imread('test5_depth.png', cv2.IMREAD_UNCHANGED)

    if depth_image is None:
        print("Error loading depth image.")
    else:
        depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

        #选择ROI
        roi = select_roi(depth_gray)

        # 计算faxian
        normal_vector = calculate_normal(depth_gray, roi)

        # 可视化
        visualize_result(depth_gray, roi, normal_vector)


