import cv2 as cv
import numpy as np

def process_frame(frames, image, lower_green, upper_green):
    '''
    Process a single frame to apply chroma key and replace green screen with an image.

    Args:
        frame: Input frame from video.
        image: Image to insert (with or without alpha channel).
        lower_green: Lower HSV threshold for green.
        upper_green: Upper HSV threshold for green.

    Returns:
        Processed frame.
    '''
    output_frames = []
    for frame in frames:
        # Convert frame to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Create mask for green screen
        mask = cv.inRange(hsv, lower_green, upper_green)  # Should be uint8 by default
        mask = mask.astype(np.uint8)  # Explicitly ensure mask is uint8
        mask_inv = cv.bitwise_not(mask)

        # Ensure image has the same size as frame
        if image.shape[:2] != frame.shape[:2]:
            image = cv.resize(image, (frame.shape[1], frame.shape[0]))

        # Extract foreground (non-green areas)
        foreground = cv.bitwise_and(frame, frame, mask=mask_inv)

        # Handle image with or without alpha channel
        if image.shape[2] == 4:  # Image has alpha channel
            img_rgb = image[:, :, :3]
            img_alpha = image[:, :, 3]
            img_mask = cv.merge([img_alpha, img_alpha, img_alpha])
            img_foreground = cv.bitwise_and(img_rgb, img_mask)
            background = cv.bitwise_and(img_rgb, cv.bitwise_not(img_mask))
            result = cv.add(foreground, img_foreground)
            result = cv.add(result, background)
        else:  # Image without alpha channel
            background = cv.bitwise_and(image, image, mask=mask)
            result = cv.add(foreground, background)

        output_frames.append(result)

    return output_frames