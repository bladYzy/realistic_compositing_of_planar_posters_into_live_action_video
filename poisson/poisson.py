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
image = cv2.imread('source1.jpg')  # 要getmask的照片 source image
mask_image = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_mask)


def select_mask(image):
    global mask_points, mask_image

    print("Draw the mask region on the image. Right-click to remove last point. Press 'ENTER' to confirm.")
    while True:
        cv2.imshow("Image", mask_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter key
            break

    # Create mask from points
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(mask_points, np.int32)
    cv2.fillPoly(mask, [points], (255))

    cv2.destroyWindow("Image")
    return mask

# Test the function
mask = select_mask(image)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



def fix_images(source, mask, target, offset):
    # Create empty arrays for the new source and mask with the size of the target
    new_source = np.zeros_like(target)
    new_mask = np.zeros_like(mask)
    
    # Calculate the bounds of the new source and mask
    y_start = max(0, offset[0])
    y_end = min(target.shape[0], offset[0] + source.shape[0])
    
    x_start = max(0, offset[1])
    x_end = min(target.shape[1], offset[1] + source.shape[1])
    
    # Calculate the bounds of the original source and mask
    source_y_start = max(0, -offset[0])
    source_y_end = source_y_start + y_end - y_start
    
    source_x_start = max(0, -offset[1])
    source_x_end = source_x_start + x_end - x_start
    
    # Place the source and mask into the new images at the offset position
    new_source[y_start:y_end, x_start:x_end] = source[source_y_start:source_y_end, source_x_start:source_x_end]
    new_mask[y_start:y_end, x_start:x_end] = mask[source_y_start:source_y_end, source_x_start:source_x_end]
    
    return new_source, new_mask, target


def poisson_blend(source, mask, target, offset):
    # Ensure the source, mask, and target are the same size
    source, mask, target = fix_images(source, mask, target, offset)
    
    rows, cols, num_colors = source.shape
    max_index = rows * cols
    
    # Initialize the sparse matrix A and vectors x and b for each color channel
    A = scipy.sparse.lil_matrix((max_index, max_index))
    b = np.zeros((max_index, num_colors))
    
    # Create an index map for pixels
    index_map = np.arange(max_index).reshape(rows, cols)
    
    # Define neighbor positions (up, left, down, right)
    positions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    
    # Loop through each color channel
    for color in range(num_colors):
        # Flatten the color channels
        source_flat = source[:, :, color].flatten()
        target_flat = target[:, :, color].flatten()
        
        # For each pixel in the mask
        for index in range(max_index):
            if mask.flat[index]:
                A[index, index] = 4
                for pos in positions:
                    i, j = divmod(index, cols)
                    ni, nj = i + pos[0], j + pos[1]
                    if 0 <= ni < rows and 0 <= nj < cols:
                        n_index = index_map[ni, nj]
                        if mask[ni, nj]:
                            A[index, n_index] = -1
                        else:
                            b[index, color] += target_flat[n_index]
                # Compute b using the source's gradients
                b[index, color] += 4 * source_flat[index] - sum(source_flat[index + offset] for offset in [1, -1, cols, -cols] if 0 <= index + offset < max_index)

    # Convert A to a more efficient format for solving
    A = A.tocsc()
    
    # Solve the system for each color channel
    result = np.zeros_like(target)
    for color in range(num_colors):
        x = spsolve(A, b[:, color])
        result_flat = result[:, :, color].flatten()
        result_flat[mask.flatten()] = x
        result[:, :, color] = result_flat.reshape(rows, cols)
        
    return result


def main():
    source_path = 'source1.jpg'
    target_path = 'target1.jpg'

    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    mask = select_mask(source)

    source, mask, target = fix_images(source, mask, target, offset=(0, 0))

    output = poisson_blend(source, mask, target)

    cv2.imwrite('output.jpg', output)
    cv2.imshow('Blended Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
