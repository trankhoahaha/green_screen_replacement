import cv2 as cv

def get_green_screen_bbox(mask):
    # Find contours of the green screen area
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    return (x, y, w, h)

def replace_green_screen03(frames, bg_image, lower_green, upper_green):
    output_frames = []

    for frame in frames:
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_frame, lower_green, upper_green)
        bbox = get_green_screen_bbox(mask)
        if bbox is None:
            output_frames.append(frame.copy())
            continue

        x, y, w, h = bbox
        bg_image_resized = cv.resize(bg_image, (w, h), interpolation=cv.INTER_AREA)
        result = frame.copy()
        roi_mask = mask[y:y+h, x:x+w]
        result[y:y+h, x:x+w][roi_mask != 0] = bg_image_resized[roi_mask != 0]
        output_frames.append(result)

    return output_frames