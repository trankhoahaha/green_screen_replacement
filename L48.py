import cv2
import numpy as np

def replace_green_screen48(frames, bg_image, lower_green, upper_green):

    output_frames = []

    # Morphology kernel & threshold
    kernel = np.ones((5,5), np.uint8)
    min_area = 100000

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Tạo mask phát hiện màu xanh
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Tìm contour từ mask gốc
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Tạo mask sạch rỗng
        clean_mask = np.zeros_like(mask)
        output = frame.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)

                # Cắt mask trong vùng box
                roi_mask = mask[y:y+h, x:x+w]

                # Morphology chỉ trong box
                roi_clean = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
                roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_DILATE, kernel)

                # Dán vào mask chính
                clean_mask[y:y+h, x:x+w] = roi_clean

                # Resize background theo box
                bg_resized = cv2.resize(bg_image, (w, h))

                # Mask cho vùng hiện tại
                box_mask = np.zeros_like(mask)
                box_mask[y:y+h, x:x+w] = roi_clean

                # Lấy phần không phải màu xanh
                mask_inv = cv2.bitwise_not(box_mask)
                fg = cv2.bitwise_and(output, output, mask=mask_inv)

                # Áp ảnh nền vào vùng mask
                bg_full = np.zeros_like(frame)
                bg_full[y:y+h, x:x+w] = bg_resized
                bg_masked = cv2.bitwise_and(bg_full, bg_full, mask=box_mask)

                # Ghép nền + gốc
                output = cv2.add(fg, bg_masked)
            
        output_frames.append(output)

    return output_frames
