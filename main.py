import numpy as np
import cv2 as cv

from L06 import replace_green_screen_adaptive
from L18 import replace_green_screen1
from L30 import replace_green_screen30


def get_video_frames(video_path):
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video")
        return []
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames


def save_video(frames, output_path, fps=30):
    height, width = frames[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print("Video saved to", output_path)


if __name__ == '__main__':
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    bg_image = cv.imread("IMG_0054.JPG")
    if bg_image is None:
        raise Exception("Failed to load background image")

    video_frames = get_video_frames("L30.mp4")
    if not video_frames:
        raise Exception("Failed to load video frames")

    output_frames = replace_green_screen30(video_frames, bg_image, lower_green, upper_green)
    save_video(output_frames, "output_video_L30.mp4")
