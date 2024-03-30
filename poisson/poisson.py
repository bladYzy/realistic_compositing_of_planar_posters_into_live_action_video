import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Load images
source_path = 'source0.jpg'
target_path = 'target0.jpg'
source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)

if source_image is None or target_image is None:
    raise ValueError("One of the images didn't load correctly. Please check the file paths.")

points = []
mask_defined = False

# Mouse callback function to capture clicks and draw the mask
def draw_mask(event, x, y, flags, param):
    global points, mask_defined, source_image

    if event == cv2.EVENT_LBUTTONDOWN and not mask_defined:
        points.append((x, y))

        # Draw the point
        cv2.circle(source_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Source Image", source_image)

        # Draw lines if more than 1 point
        if len(points) > 1:
            cv2.line(source_image, points[-2], points[-1], (255, 0, 0), 2)
            cv2.imshow("Source Image", source_image)

        # If four points are selected, draw the polygon and set the mask_defined flag
        if len(points) == 4:
            cv2.fillPoly(source_image, [np.array(points)], (0, 255, 0))
            mask_defined = True
            cv2.imshow("Source Image", source_image)
            cv2.waitKey(500)  # Wait 500 ms before closing the window

# Create a window and set the mouse callback function
cv2.namedWindow("Source Image")
cv2.setMouseCallback("Source Image", draw_mask)

# Show the source image and wait until the user has defined the mask
while not mask_defined:
    cv2.imshow("Source Image", source_image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()

# Check if the mask is defined, if not, exit
if not mask_defined:
    raise ValueError("Mask not defined.")

# Create the mask using the points selected by the user
mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [np.array(points)], (255))

# Offset of where to place the top-left corner of the source image on the target
offset_x, offset_y = 50, 100

# The following code defines the Poisson blending function
def poisson_blend(source, mask, target, offset_x, offset_y):
    # Compute regions of interest
    y_max, x_max = mask.shape
    region_source = (
        max(-offset_y, 0),
        min(target.shape[0] - offset_y, y_max),
        max(-offset_x, 0),
        min(target.shape[1] - offset_x, x_max),
    )
    region_target = (
        max(offset_y, 0),
        min(target.shape[0], offset_y + y_max),
        max(offset_x, 0),
        min(target.shape[1], offset_x + x_max),
    )
    
    # Masks of region in which to blend
    mask_target = mask[
        region_source[0] : region_source[1], region_source[2] : region_source[3]
    ]
    mask_indices = np.where(mask_target.flatten())[0]
    
    # Laplacian operator for a single channel
    laplacian = scipy.sparse.diags([4, -1, -1, -1, -1], [0, -1, 1, -y_max, y_max], shape=(y_max * x_max, y_max * x_max))
    
    # For each layer (channel) in the image
    for channel in range(source.shape[2]):
        # Take one channel of each image
        source_layer = source[:, :, channel]
        target_layer = target[:, :, channel]
        
        # Create the masked source and target images
        source_region = source_layer[
            region_source[0] : region_source[1], region_source[2] : region_source[3]
        ]
        target_region = target_layer[
            region_target[0] : region_target[1], region_target[2] : region_target[3]
        ]

        # Create the matrix representing the blending mask
        mask_flat = mask_target.flatten()
        mask_matrix = scipy.sparse.diags(mask_flat)
        
        # Create the known part of the Poisson equation
        laplacian_masked = mask_matrix @ laplacian
        target_flat = target_region.flatten()
        boundary_values = laplacian_masked @ target_flat
        
        # Solve the Poisson equation
        region_flat = region_source[0] * x_max + region_source[2]
        region_shape = (region_source[1] - region_source[0], region_source[3] - region_source[2])
        mask_flat_region = mask_flat[region_flat : region_flat + np.prod(region_shape)]
        laplacian_region = laplacian[region_flat : region_flat + np.prod(region_shape), region_flat : region_flat + np.prod(region_shape)]
        
        # Solve for masked region
        masked_region_flat = spsolve(laplacian_region, boundary_values[mask_flat_region])
        masked_region = np.reshape(masked_region_flat, region_shape)
        
        # Place solved region into the target image
        target_layer[
            region_target[0] : region_target[1], region_target[2] : region_target[3]
        ][mask_target == 255] = masked_region[mask_target == 255]
        target[:, :, channel] = target_layer
    
    return target

# Perform Poisson blending
result = poisson_blend(source_image, mask, target_image, offset_x, offset_y)

# Save the output image
output_path = 'blended_output.jpg'
cv2.imwrite(output_path, result)

# Display the output image
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Blended Image')
plt.show()