import cv2
import numpy as np
from sklearn.cluster import KMeans
import ctypes

def get_screen_size():
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    return screen_width, screen_height

def set_window(name, image_shape):
    screen_width, screen_height = get_screen_size()
    scale = min(screen_width / image_shape[1], screen_height / image_shape[0], 1)
    window_width = int(image_shape[1] * scale)
    window_height = int(image_shape[0] * scale)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_width, window_height)

def segment_planes(normals, n_clusters=3):
    reshaped_normals = normals.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reshaped_normals)
    labels = kmeans.labels_.reshape(normals.shape[0], normals.shape[1])
    return labels, kmeans.cluster_centers_

def display_segmented_planes(image, labels, cluster_centers):
    unique_labels = np.unique(labels)
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3), dtype=np.uint8)
    segmented_image = colors[labels]
    set_window("Segmented Planes", image.shape)
    cv2.imshow("Segmented Planes", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return colors

def select_plane_and_get_normal(image, labels, cluster_centers, colors):
    print("Select a plane by clicking on it in the window.")
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            label = labels[y, x]
            print(f"Selected Plane Normal: {cluster_centers[label]}")
            color = colors[label]
            selected_region = np.all(colors[labels] == color, axis=-1)
            set_window("Selected Plane", image.shape)
            cv2.imshow("Selected Plane", selected_region.astype(np.uint8) * 255)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    segmented_image = colors[labels]
    set_window("Select Plane", image.shape)
    cv2.setMouseCallback("Select Plane", mouse_callback)
    cv2.imshow("Select Plane", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image_path, normal_map_path):
    image = cv2.imread(image_path)
    normals = cv2.imread(normal_map_path, cv2.IMREAD_COLOR).astype(np.float32) / 127.5 - 1
    labels, cluster_centers = segment_planes(normals, n_clusters=5)
    colors = display_segmented_planes(image, labels, cluster_centers)
    select_plane_and_get_normal(image, labels, cluster_centers, colors)



if __name__ == "__main__":
    image_path = 'test2/test2.jpg'
    normal_map_path = 'test2/test2_normal_resize.png'
    main(image_path, normal_map_path)
