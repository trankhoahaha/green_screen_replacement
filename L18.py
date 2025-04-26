import cv2 as cv

def replace_green_screen1(frames, replacement_img, lower_green, upper_green):
    result_frames = []
    for frame in frames:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_green, upper_green)

        # Tìm bounding box của vùng xanh
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)

            # Resize ảnh thay thế theo vùng xanh
            resized_img = cv.resize(replacement_img, (w, h))

            # Tạo mặt nạ nghịch và ghép ảnh
            roi = frame[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w]
            mask_inv = cv.bitwise_not(mask_roi)

            bg = cv.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv.bitwise_and(resized_img, resized_img, mask=mask_roi)

            combined = cv.add(bg, fg)
            frame[y:y+h, x:x+w] = combined

        result_frames.append(frame)
    return result_frames