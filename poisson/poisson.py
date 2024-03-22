import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve

# Global variables to store mask points and the mask image
mask_points = []
mask_image = None

def draw_mask(event, x, y, flags, param):
    global mask_points, mask_image
    if event == cv2.EVENT_LBUTTONDOWN:
        mask_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and mask_points:
        mask_points.pop()
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        mask_points.append((x, y))

    # Update the mask image
    if mask_points:
        mask_image = np.copy(param)  # Use the passed image
        cv2.polylines(mask_image, [np.array(mask_points, np.int32)], False, (0, 255, 0), 1)
        cv2.imshow("Image", mask_image)

def select_mask(image):
    global mask_points, mask_image
    mask_points = []  # Reset the points for a new mask
    mask_image = np.copy(image)  # Reset the mask image
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_mask, image)  # Pass the image as a parameter

    print("Draw the mask region on the image. Right-click to remove the last point. Press 'ENTER' to confirm.")
    while True:
        cv2.imshow("Image", mask_image)
        if cv2.waitKey(1) & 0xFF in [13, 27]:  # Enter or ESC
            break
    cv2.destroyAllWindows()

    # Create the mask based on the drawn region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if mask_points:
        cv2.fillPoly(mask, [np.array(mask_points, np.int32)], 255)
    return mask

# Function to adjust the images and mask to align them correctly
def fix_images(source, mask, target, offset):
    y_offset, x_offset = offset
    source_height, source_width = source.shape[:2]
    target_height, target_width = target.shape[:2]
    new_source = np.zeros_like(target)
    new_mask = np.zeros(target.shape[:2], dtype=np.uint8)

    # Calculating the region of interest in both source and target images
    y_start = max(0, y_offset)
    x_start = max(0, x_offset)
    y_end = min(target_height, y_offset + source_height)
    x_end = min(target_width, x_offset + source_width)

    # Adjusting the source image and mask to fit into the target image
    source_y_start = max(0, -y_offset)
    source_x_start = max(0, -x_offset)
    source_y_end = source_height - max(0, (y_offset + source_height) - target_height)
    source_x_end = source_width - max(0, (x_offset + source_width) - target_width)

    # Placing the source image and mask onto the target image based on the calculated region
    if y_end > y_start and x_end > x_start:
        new_source[y_start:y_end, x_start:x_end] = source[source_y_start:source_y_end, source_x_start:source_x_end]
        new_mask[y_start:y_end, x_start:x_end] = mask[source_y_start:source_y_end, source_x_start:source_x_end]

    return new_source, new_mask, target

def poisson_blend(source, mask, target, offset):
    # Preprocess images and mask
    source_processed, mask_processed, target_processed = fix_images(source, mask, target, offset)
    rows, cols, _ = target_processed.shape
    N = rows * cols
    mask_flat = mask_processed.flatten()

    # Initialize sparse matrix A and vectors b for each color channel
    A = scipy.sparse.lil_matrix((N, N))
    b = np.zeros((N, 3))

    # Utility function to convert 2D indices to 1D
    def index(i, j):
        return i * cols + j

    # Define neighbor positions (up, left, down, right)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for color in range(3):
        # Flatten the current channel for source and target images
        source_flat = source_processed[:, :, color].flatten()
        target_flat = target_processed[:, :, color].flatten()

        # Iterate through each pixel in the mask
        for i in range(rows):
            for j in range(cols):
                idx = index(i, j)
                if mask_flat[idx]:  # Pixel is under the mask
                    A[idx, idx] = 4  # Diagonal value is 4

                    # Process neighbors
                    for dy, dx in neighbors:
                        ny, nx = i + dy, j + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            n_idx = index(ny, nx)
                            if mask_flat[n_idx]:  # Neighbor is also in the mask
                                A[idx, n_idx] = -1
                            else:
                                # Boundary condition: use target image value
                                b[idx, color] += target_flat[n_idx]

                    # Compute the Laplacian operator (discrete Laplace)
                    for dy, dx in neighbors:
                        ny, nx = i + dy, j + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            n_idx = index(ny, nx)
                            b[idx, color] += source_flat[idx] - source_flat[n_idx]

    # Convert A to CSR format for efficient solving
    A_csr = A.tocsr()

    # Solve the linear system A * x = b for each color channel
    result = np.copy(target_processed)  # Initialize result image
    for color in range(3):
        x = spsolve(A_csr, b[:, color])
        result_flat = result[:, :, color].flatten()
        result_flat[mask_flat] = x  # Update only pixels within the mask
        result[:, :, color] = result_flat.reshape((rows, cols))

    return result


def main():
    source_path = "source0.jpg"
    target_path = "target0.jpg"

    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    mask = select_mask(source)

    # Define offset from the top left corner (y, x)
    offset = (50, 50)  # Example offset, update according to your needs

    # Perform Poisson blending
    blended_result = poisson_blend(source, mask, target, offset)

    # Show and save the blended image
    cv2.imshow('Blended Image', blended_result)
    cv2.imwrite('blended_result.jpg', blended_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()