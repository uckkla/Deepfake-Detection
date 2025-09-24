import cv2
import os
from mtcnn import MTCNN

def extract_faces(frame_dir, output_dir, min_confidence=0.95, margin=0.3, filename_prefix="", single_image=False):
    os.makedirs(output_dir, exist_ok=True)
    detector = MTCNN()

    if single_image:
        frame_paths = [frame_dir]  # full path to the image
    else:
        frame_paths = [os.path.join(frame_dir, fname) for fname in sorted(os.listdir(frame_dir))]

    # Loop through each frame in dir
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"[WARNING] Could not read image: {frame_path}")
            continue
        
        faces = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for i, face in enumerate(faces):
            # Skip face if low confidence
            if face["confidence"] < min_confidence:
                continue

            x, y, w, h = face["box"]
            
            # Get boundary of face, margin added to not miss any outside facial features
            x1 = max(0, int(x - (margin * w)))
            y1 = max(0, int(y - (margin * h)))
            x2 = min(frame.shape[1], int(x + w + (margin * w)))
            y2 = min(frame.shape[0], int(y + h + (margin * h)))

            # Set boundaries of face
            face_img = frame[y1:y2, x1:x2]

            # Output face
            frame_name = os.path.basename(frame_path)
            output_path = os.path.join(output_dir, f"{filename_prefix}_{os.path.splitext(frame_name)[0]}_face_{i}.jpg")
            cv2.imwrite(output_path, face_img)
