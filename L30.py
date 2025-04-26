import cv2 as cv

def is_rectangular(contour, aspect_range=(0.45, 0.6), min_area=5000, rectangularity_thresh=0.85):
    x, y, w, h = cv.boundingRect(contour)
    area = cv.contourArea(contour)
    aspect_ratio = w / h if h != 0 else 0

    # Diện tích của hình chữ nhật bao contour
    rect_area = w * h
    rectangularity = area / rect_area if rect_area != 0 else 0

    # Điều kiện là vùng chữ nhật rõ (màn hình)
    if min_area < rect_area and aspect_range[0] < aspect_ratio < aspect_range[1] and rectangularity > rectangularity_thresh:
        return True
    return False

def replace_green_screen30(frames, replacement_img, lower_green, upper_green):
    result_frames = []

    for i, frame in enumerate(frames):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_green, upper_green)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        selected_bbox = None
        for contour in contours:
            if is_rectangular(contour):
                x, y, w, h = cv.boundingRect(contour)
                selected_bbox = (x, y, w, h)
                break  # lấy contour đầu tiên hợp lệ (có thể cải tiến xếp hạng thêm nếu cần)

        if selected_bbox:
            x, y, w, h = selected_bbox
            resized_img = cv.resize(replacement_img, (w, h))

            roi = frame[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w]
            mask_inv = cv.bitwise_not(mask_roi)

            bg = cv.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv.bitwise_and(resized_img, resized_img, mask=mask_roi)

            combined = cv.add(bg, fg)
            frame[y:y+h, x:x+w] = combined

        result_frames.append(frame)
    return result_frames

