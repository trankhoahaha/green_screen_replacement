import cv2 as cv
from L06 import detect_green_screen

def check_screen_not_full(frames, lower_green, upper_green):
    not_full_green = False
    area_threshold=0.05
    max_area = 0
    for frame in frames[:4]:
        box, mask = detect_green_screen(frame, lower_green, upper_green)
        if box is None:
            continue
        x, y, w, h = box
        print(x, y, w, h)
        print(frame.shape[1], frame.shape[0])
        # check 4 corner of box is not 0
        if x > 0 and y > 0 and x + h < frame.shape[1] and y + w < frame.shape[0]:
            print("full green")
            break
        area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        if area / frame_area > area_threshold and area > max_area:
            not_full_green = True
            break
    return not_full_green


def is_rectangular(frame, lower_green, upper_green, aspect_range=(0.45, 0.6), min_area=5000, rectangularity_thresh=0.85):

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_green, upper_green)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
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