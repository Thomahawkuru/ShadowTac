import cv2
import numpy as np

def crop_image(image, radius):
    # Find the center of the image
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    
    # Define the radius of the circle
    # Create a mask with a circular shape
    # image = image[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, (255), -1)
    
    # Apply the mask to the sub image
    image = cv2.bitwise_and(image, image, mask=mask)

    mask = np.ones_like(image)*1
    cv2.circle(mask, (center_x, center_y), radius, (0,0,0), -1)
    
    image = cv2.add(image, mask)

    return image

def crop_image_small(image, radius):
    # Find the center of the image
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    
    # Define the radius of the circle
    # Create a mask with a circular shape
    image = image[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
    return image
