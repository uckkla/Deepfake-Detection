import cv2
import os

def extract_frames(video_path, output_folder, fps=5, min_width=300):
    """Extracts frames from video with quality controls.
    
    Args:
        video_path: Input video file path
        output_folder: Directory to save frames
        fps: Target frames per second
        min_width: Minimum width to not upscale
    Returns:
        Number of frames saved
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Failed to open video file")

    native_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(native_fps/fps)

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if needed
    saved_count = 0

    print(f"Extracting {total_frames//frame_interval} frames from {video_path}...")

    while(capture.isOpened()):
        frame_num = capture.get(cv2.CAP_PROP_POS_FRAMES)
        flag, frame = capture.read()
        if not flag:
            break

        if frame_num % frame_interval == 0:
            width = frame.shape[1]
            # Enlarge image if too small for facial extraction
            if width < min_width:
                scale = min_width / width
                # Not tested other interpolations eg. INTER_LANCZOS4 and INTER_LINEAR
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # jpg significantly smaller - quicker to process. could change to png if compression is an issue
            cv2.imwrite(f"{output_folder}/frame_{saved_count:05d}.jpg", frame)
            saved_count += 1
        
    capture.release()
    print(f"Saved {saved_count} frames to {output_folder}")
    return saved_count