import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve


# Initialize the list to store mask points
mask_points = []

# Callback function for mouse events
def draw_mask(event, x, y, flags, param):
    global mask_points, mask_image

    if event == cv2.EVENT_LBUTTONDOWN:
        mask_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        mask_points.pop()

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
            mask_points.append((x, y))

    # Redraw the mask
    if len(mask_points) > 0:
        mask_image[:] = image
        pts = np.array(mask_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mask_image, [pts], False, (0, 255, 0), 1)
        cv2.imshow("Image", mask_image)

# Initialize mask image
image = cv2.imread('source0.jpg')  # 要getmask的照片 source image
mask_image = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_mask)


def select_mask(image):
    global mask_points, mask_image

    print("Draw the mask region on the image. Right-click to remove last point. Press 'ENTER' to confirm.")
    while True:
        cv2.imshow("Image", mask_image)
        key = cv2.waitKey(1) & 0xFF
        print(f"Key pressed: {key}")  # Debug print

        if key == 13 or key == 10:  # Enter key (sometimes 10 on certain systems)
            print("Enter key detected, breaking loop.")  # Debug print
            break
        elif key == 27:  # ESC key to exit loop in case 'Enter' isn't working
            print("ESC key detected, breaking loop.")  # Debug print
            break

    # Create mask from points
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(mask_points, np.int32)
    cv2.fillPoly(mask, [points], (255))

    cv2.destroyWindow("Image")
    return mask



def fix_images(source, mask, target, offset):
    # Get the dimensions of source and target images
    source_height, source_width = source.shape[:2]
    target_height, target_width = target.shape[:2]

    # Initialize new_source and new_mask arrays
    new_source = np.zeros_like(target)
    new_mask = np.zeros(target.shape[:2], dtype=np.uint8)
    
    # Calculate the starting and ending coordinates for source and target
    y_start_target, x_start_target = max(0, offset[0]), max(0, offset[1])
    y_end_target = min(target_height, offset[0] + source_height)
    x_end_target = min(target_width, offset[1] + source_width)

    y_start_source = max(0, -offset[0])
    x_start_source = max(0, -offset[1])
    y_end_source = source_height - max(0, (offset[0] + source_height) - target_height)
    x_end_source = source_width - max(0, (offset[1] + source_width) - target_width)

    # Ensure the regions to be copied are valid
    if y_end_target - y_start_target > 0 and x_end_target - x_start_target > 0:
        # Copy the relevant parts of source and mask to the target canvas
        new_source[y_start_target:y_end_target, x_start_target:x_end_target] = \
            source[y_start_source:y_end_source, x_start_source:x_end_source]
        new_mask[y_start_target:y_end_target, x_start_target:x_end_target] = \
            mask[y_start_source:y_end_source, x_start_source:x_end_source]

    return new_source, new_mask, target




def poisson_blend(source, mask, target, offset):
    # Preprocess images and mask
    source_processed, mask_processed, target_processed = fix_images(source, mask, target, offset)
    
    rows, cols, _ = target_processed.shape
    N = rows * cols
    mask_flat = mask_processed.flatten()
    
    # Initialize sparse matrix and vectors for b
    A = scipy.sparse.lil_matrix((N, N))
    b = np.zeros(N)
    
    for color in range(3):  # Iterate through color channels
        source_channel = source_processed[:, :, color].flatten()
        target_channel = target_processed[:, :, color].flatten()
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if mask_flat[idx]:  # If pixel is part of the mask
                    A[idx, idx] = 4
                    for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighbors
                        ni, nj = i + d[0], j + d[1]
                        n_idx = ni * cols + nj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if mask_flat[n_idx]:
                                A[idx, n_idx] = -1
                            else:
                                b[idx] += target_channel[n_idx]
                        # Set b for the source's gradient
                        b[idx] += source_channel[idx] - source_channel[n_idx] if mask_flat[n_idx] else 0
        
        # Solve the linear system
        A_csr = A.tocsr()
        x = spsolve(A_csr, b)
        
        # Update the target image with the solution
        target_processed[:, :, color] = x.reshape((rows, cols))
    
    return target_processed


def main():
    source_path = "source0.jpg"
    target_path = "target0.jpg"
    mask_path = "mask.jpg"
    
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    mask = cv2.imread(mask_path, 0)  # Assuming mask is grayscale
    mask = mask / 255  # Normalize mask to be 0 or 1
    
    offset = (50, 50)  # Example offset
    
    blended = poisson_blend(source, mask, target, offset)
    cv2.imshow("Blended Image", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
