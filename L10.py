import cv2 as cv
import numpy as np
from collections import deque

def process_frame(frames, background, lower_green, upper_green):
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
    green_region_buffer = deque(maxlen=30)
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]

    for idx, frame in enumerate(frames):
        print(idx)

        # Step 1: Create green screen mask
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_green, upper_green)
        mask = mask.astype(np.uint8)

        # Refine mask to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.erode(mask, kernel, iterations=1)
        mask = cv.dilate(mask, kernel, iterations=2)
        mask_inv = cv.bitwise_not(mask)

        # Step 2: Find green screen region
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        green_region = None
        if contours:
            # Get the largest contour (assumed to be the green screen)
            largest_contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest_contour)
            if area > 0.1 * frame_width * frame_height:  # Ensure significant size

                x, y, w, h = cv.boundingRect(largest_contour)
                # remove green region if w or h is too small => noise
                if w > 10 and h > 10:
                    roi_mask = mask[y:y+h, x:x+w]
                    # Phát hiện đường thẳng trong vùng bounding box
                    edges = cv.Canny(roi_mask, 50, 150)
                    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)
                    if lines is not None and len(lines) >= 2:

                        green_region = (x, y, w, h)
                        green_region_buffer.append(green_region)

        # Smooth green_region using buffer
        if green_region_buffer:
            # Average the region coordinates
            x_sum, y_sum, w_sum, h_sum = 0, 0, 0, 0
            for x, y, w, h in green_region_buffer:
                x_sum += x
                y_sum += y
                w_sum += w
                h_sum += h
            n = len(green_region_buffer)
            green_region = (int(x_sum / n), int(y_sum / n), int(w_sum / n), int(h_sum / n))
        else:
            green_region = (0, 0, frame_width, frame_height)  # Fallback to full frame

        x, y, w, h = green_region

        # Step 3: Resize background to green screen region
        bg_resized = cv.resize(background, (w, h), interpolation=cv.INTER_AREA)

        # Step 4: Create output frame
        result = frame.copy()
        roi = result[y:y+h, x:x+w]
        roi_foreground = cv.bitwise_and(roi, roi, mask=mask_inv[y:y+h, x:x+w])
        roi_background = cv.bitwise_and(bg_resized, bg_resized, mask=mask[y:y+h, x:x+w])
        result[y:y+h, x:x+w] = cv.add(roi_foreground, roi_background)
        output_frames.append(result)

    return output_frames