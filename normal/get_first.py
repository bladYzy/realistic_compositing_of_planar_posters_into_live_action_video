import cv2
from PIL import Image
import numpy as np


def resize_first_frame(video_path, output_image_path, target_size=(384, 384)):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        print("Error: Could not read the first frame from video.")
        return
    cap.release()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    #img_resized = img.resize(target_size, Image.BILINEAR)
    #img_resized.save(output_image_path)

    img.save(output_image_path)
    print(f"Resized first frame saved to {output_image_path}")

video_path = 'test3/test3.mp4'
output_image_path = 'test3/test3_frame.jpg'
resize_first_frame(video_path, output_image_path)
