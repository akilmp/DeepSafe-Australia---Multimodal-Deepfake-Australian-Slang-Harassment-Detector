import os
import subprocess
import tempfile
from pathlib import Path

import cv2


def anonymise_video(input_path: str, output_path: str) -> None:
    """Blur faces in a video using Haar cascades."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_video = tmp.name
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (51, 51), 0)
            frame[y:y+h, x:x+w] = blurred
        writer.write(frame)

    cap.release()
    writer.release()

    # Combine processed video with original audio
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        temp_video,
        "-i",
        input_path,
        "-map",
        "0:v",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(temp_video)


def generate_deepfake(src_video: str, tgt_face: str, output_path: str) -> None:
    """Generate a deepfake by swapping tgt_face into src_video using faceswap CLI."""
    cmd = [
        "faceswap",
        "-i",
        src_video,
        "-o",
        output_path,
        "-t",
        tgt_face,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    for video in raw_dir.glob("*.mp4"):
        output = video.with_name(video.stem + "_anon.mp4")
        anonymise_video(str(video), str(output))
