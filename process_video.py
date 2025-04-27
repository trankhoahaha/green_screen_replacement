import cv2 as cv
from L06 import get_reference_size, detect_green_screen
from L03 import get_green_screen_bbox
from check_frame import is_rectangular, check_screen_not_full


def replace_green_screen(frames, bg_image, lower_green, upper_green):

    # checking variables region
    f_not_full_green = False
    f_is_rectangular = False
    f_multiple_screen = False

    # init variables
    output_frames = []
    bbox = None
    ref_box = None #TH: màn hình bị khuất


    # checking
    f_is_rectangular = is_rectangular(frames[0], lower_green, upper_green)

    # check if screen is not full
    if check_screen_not_full(frames, lower_green, upper_green):
        f_not_full_green = True
        ref_box = get_reference_size(frames, lower_green, upper_green)

        ref_x, ref_y, ref_w, ref_h = ref_box
        bg_resized = cv.resize(bg_image, (ref_w, ref_h))
        print("Screen is not full")




    # process frames
    for idx, frame in enumerate(frames):
        print(idx)
        if f_not_full_green is True:
            box, mask = detect_green_screen(frame, lower_green, upper_green)
            if box is None:
                print("Box is None")
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
        elif f_is_rectangular is True:

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower_green, upper_green)
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                print(x, y, w, h)
                break
            
            resized_img = cv.resize(bg_image, (w, h))

            roi = frame[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w]
            mask_inv = cv.bitwise_not(mask_roi)

            bg = cv.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv.bitwise_and(resized_img, resized_img, mask=mask_roi)

            combined = cv.add(bg, fg)
            frame[y:y+h, x:x+w] = combined
            output_frames.append(frame)

    return output_frames