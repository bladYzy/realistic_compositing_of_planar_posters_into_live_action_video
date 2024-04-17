import cv2
import sys


def resize_image(image_path, target_size):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return None

    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img


if __name__ == "__main__":

    image_path = "test3/test3_frame_resize_normal.png"
    target_size = (720, 1280)
    #target_size = (384, 384)

    resized_image = resize_image(image_path, target_size)
    if resized_image is not None:
        cv2.imshow('Resized Image', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("test3/test3_normal_resize.png", resized_image)
