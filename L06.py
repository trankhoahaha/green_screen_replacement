import cv2 as cv

def detect_green_screen(frame, lower_green, upper_green):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_green, upper_green)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    max_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(max_contour)
    return (x, y, w, h), mask


def get_reference_size(frames, lower_green, upper_green, area_threshold=0.05):
    max_area = 0
    ref_box = None
    idx = 0
    total_frame = len(frames)
    for idx in range(total_frame):
        frame = frames[idx]
        box, mask = detect_green_screen(frame, lower_green, upper_green)
        if box is None:
            continue
        x, y, w, h = box

        # check break for x and y is not 0
        if x > 0 and y > 0 and x + w < frame.shape[1] and y + h < frame.shape[0]:
            break
        area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        if area / frame_area > area_threshold and area > max_area:
            max_area = area
            ref_box = box
        idx += 8
    return ref_box


def replace_green_screen_adaptive(frames, bg_image, lower_green, upper_green):
    ref_box = get_reference_size(frames, lower_green, upper_green)
    if ref_box is None:
        raise Exception("Could not detect reference green screen size")

    ref_x, ref_y, ref_w, ref_h = ref_box
    bg_resized = cv.resize(bg_image, (ref_w, ref_h))
    output_frames = []

    for idx, frame in enumerate(frames):
        print(idx)

        box, mask = detect_green_screen(frame, lower_green, upper_green)
        if box is None:
            output_frames.append(frame.copy())
            continue

        x, y, w, h = box
        result = frame.copy()

        # Crop từ ảnh nền sao cho phần viền phải hiện ra khi vùng xanh lệch phải
        offset = x - ref_x
        start_x = ref_w - w - offset  # Căn chỉnh từ bên phải
        start_x = max(0, min(ref_w - w, start_x))  # Clamp

        bg_crop = bg_resized[:, start_x:start_x + w]

        if bg_crop.shape[0] != h or bg_crop.shape[1] != w:
            bg_crop = cv.resize(bg_crop, (w, h))

        result[y:y + h, x:x + w][mask[y:y + h, x:x + w] == 255] = bg_crop[mask[y:y + h, x:x + w] == 255]
        output_frames.append(result)

    return output_frames