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
    source_height, source_width = source.shape[:2]
    target_height, target_width = target.shape[:2]

    # Create empty arrays for the new source and mask with the size of the target
    new_source = np.zeros_like(target)
    new_mask = np.zeros(target.shape[:2], dtype=np.uint8)
    
    # Calculate the overlapping region considering the offset
    y_start, x_start = offset
    y_end = min(y_start + source_height, target_height)
    x_end = min(x_start + source_width, target_width)
    
    # Calculate the bounds of the source to be used
    source_y_start = max(0, -offset[0])
    source_y_end = source_height - max(0, (y_start + source_height) - target_height)
    
    source_x_start = max(0, -offset[1])
    source_x_end = source_width - max(0, (x_start + source_width) - target_width)
    
    # Ensure there is an overlap; if not, return original images
    if y_end - y_start <= 0 or x_end - x_start <= 0:
        return source, mask, target

    # Place the source and mask into the new images at the offset position
    new_source[y_start:y_end, x_start:x_end] = source[source_y_start:source_y_end, source_x_start:source_x_end]
    new_mask[y_start:y_end, x_start:x_end] = mask[source_y_start:source_y_end, source_x_start:source_x_end]
    
    return new_source, new_mask, target



def poisson_blend(source, mask, target, offset):
    # Ensure the source, mask, and target are correctly prepared
    source, mask, target = fix_images(source, mask, target, offset)
    
    rows, cols, num_colors = target.shape
    N = rows * cols  # Total number of pixels

    # Initialize the sparse matrix A and vectors b for each color channel
    A = scipy.sparse.lil_matrix((N, N))
    b = np.zeros((N, num_colors))

    # For convenience, define the index function for the flattened array
    index = lambda i, j: i * cols + j

    # Define neighbor offsets for up, down, left, right
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    mask_flat = mask.ravel()  # Flatten the mask for easy indexing

    for color in range(num_colors):
        source_flat = source[:, :, color].ravel()
        target_flat = target[:, :, color].ravel()

        # Iterate over each pixel in the image
        for i in range(rows):
            for j in range(cols):
                idx = index(i, j)  # Linear index in the flattened array

                if mask_flat[idx]:  # If this pixel is in the mask
                    A[idx, idx] = 4  # Diagonal value

                    # Iterate over all neighbors
                    for dy, dx in offsets:
                        ny, nx = i + dy, j + dx

                        # Boundary check
                        if 0 <= ny < rows and 0 <= nx < cols:
                            n_idx = index(ny, nx)

                            if mask_flat[n_idx]:  # If neighbor is also in the mask
                                A[idx, n_idx] = -1
                            else:
                                # For boundary pixels, use target image values
                                b[idx, color] += target_flat[n_idx]
                    
                    # Set up b using the divergence of the gradient field
                    # This part depends on your specific implementation needs
                    # For simplicity, let's use the source image intensity for now
                    b[idx, color] += 4 * source_flat[idx] - \
                                     sum(source_flat[index(i + y, j + x)] for y, x in offsets if 0 <= i + y < rows and 0 <= j + x < cols)

    # Convert A to CSR format for efficient solving
    A_csr = A.tocsr()

    # Solve the system A * x = b for each color channel
    result = np.copy(target)  # Start with the target as a base for the result

    for color in range(num_colors):
        x = spsolve(A_csr, b[:, color])  # Solve for this color channel

        # Place the solved values back into the image
        result_flat = result[:, :, color].ravel()
        result_flat[mask_flat] = x  # Only update pixels within the mask
        result[:, :, color] = result_flat.reshape((rows, cols))

    return result


def main():
    source_path = 'source0.jpg'
    target_path = 'target0.jpg'

    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    mask = select_mask(source)

    source, mask, target = fix_images(source, mask, target, offset=(100, 100))

    output = poisson_blend(source, mask, target, offset=(100, 100))

    cv2.imwrite('output.jpg', output)
    cv2.imshow('Blended Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
