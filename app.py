from flask import Flask, render_template, request, Response
import cv2 as cv
import os
import tempfile
from L30 import replace_green_screen30
from const import lower_green, upper_green

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        bg_file = request.files['background']

        if not video_file or not bg_file:
            return "Missing file(s)", 400

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        bg_path = os.path.join(app.config['UPLOAD_FOLDER'], bg_file.filename)
        video_file.save(video_path)
        bg_file.save(bg_path)

        bg_image = cv.imread(bg_path)
        if bg_image is None:
            return "Failed to load background image", 500

        video_frames = get_video_frames(video_path)
        if not video_frames:
            return "Failed to load video frames", 500

        output_frames = replace_green_screen30(video_frames, bg_image, lower_green, upper_green)

        # Encode video vào bộ nhớ RAM
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        height, width = output_frames[0].shape[:2]
        fourcc = cv.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv.VideoWriter(temp_video.name, fourcc, 30, (width, height))

        for frame in output_frames:
            out.write(frame)
        out.release()

        def generate():
            with open(temp_video.name, 'rb') as f:
                data = f.read(1024)
                while data:
                    yield data
                    data = f.read(1024)
            os.unlink(temp_video.name)  # Xóa file tạm sau khi trả xong

        return Response(generate(), mimetype='video/mp4')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
