import json
import os
from face_extraction import extract_faces
from frame_extraction import extract_frames

def process_dfdc_videos(video_dir, output_base_dir, target_fps=5):
    """
    Processing videos provided by DFDC (Deepfake Detection Challenge), Meta
    Source: https://www.kaggle.com/c/deepfake-detection-challenge/data
    Args:
        video_path: Input video file path
        output_base_dir: Directory containing dataset
        target_fps: Target frames per second
    """
    # Metadata contains data of all videos - if they are fake or real
    with open(os.path.join(video_dir, "metadata.json")) as metadata_json:
        metadata = json.load(metadata_json)
    
    real_dir, fake_dir, temp_dir = create_required_directories(output_base_dir)
    
    for filename, data in metadata.keys():
        video_path = os.path.join(video_dir, filename)
        if not os.path.exists(video_path):
            print(f"Warning: Missing video file {filename}")
            continue
        
        output_dir = real_dir if data["label"] == "REAL" else fake_dir

        process_video(
            video_path=video_path,
            output_dir=output_dir,
            temp_dir=temp_dir,
            target_fps=target_fps
        )

def process_celeb_df_videos(video_dir, output_base_dir, target_fps=5):
    """
    Processing videos provided by Celeb-DF-V2
    Source: https://github.com/yuezunli/celeb-deepfakeforensics
    Args:
        video_path: Input video file path
        output_base_dir: Directory containing dataset
        target_fps: Target frames per second
    """
    real_dir, fake_dir, temp_dir = create_required_directories(output_base_dir)

    # Process real videos (Celeb-real and YouTube-real)
    for real_subdir in ["Celeb-real", "YouTube-real"]:
        real_path = os.path.join(video_dir, real_subdir)
        if not os.path.exists(real_path):
            print(f"Warning: Missing directory {real_subdir}")
            continue

        for video_file in os.listdir(real_path):
            process_video(
                video_path=os.path.join(real_path, video_file),
                output_dir=real_dir,
                temp_dir=temp_dir,
                target_fps=target_fps
            )
    
    # Process fake videos (Celeb-synthesis)
    fake_path = os.path.join(video_dir, "Celeb-synthesis")
    if not os.path.exists(fake_path):
        print(f"Warning: Missing directory {fake_path}")
    else:
        for video_file in os.listdir(fake_path):
            process_video(
                video_path=os.path.join(fake_path, video_file),
                output_dir=fake_dir,
                temp_dir=temp_dir,
                target_fps=target_fps
            )

def process_video(video_path, output_dir, temp_dir, target_fps):
    """
    Generic processing of videos - used for datasets and user input
    Args:
        video_path: Input video file path
        output_dir: Directory to save faces
        temp_dir: Directory to temporarily save frames
        target_fps: Target frames per second
    """
    try:
        extract_frames(
            video_path=video_path,
            output_folder=temp_dir,
            fps=target_fps
        )

        video_id = os.path.splitext(os.path.basename(video_path))[0]
        extract_faces(
            frame_dir=temp_dir,
            output_dir=output_dir,
            filename_prefix=video_id
            )
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
    """
    # Commented out temporarily in case of mistakes, removes each file from temp
    finally:
        if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
    """

"""
# Need to change logic - if image should be deleted here or somewhere else
def process_image(image_dir, output_dir, temp_dir):
    #
    Generic processing of image - used for user input
    Args:
        image_path: Input image file path
        output_dir: Directory to save faces
    #
    image_path = os.path.join(image_dir, os.listdir(image_dir))
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    extract_faces(
        frame_dir=image_dir,
        output_dir=output_dir,
        filename_prefix=image_id
    )
    os.remove(image_path)
"""

def create_required_directories(output_base_dir):
    """
    Creates the real, fake, and temp directories required for processing the datasets.
    Args:
        output_base_dir: Directory where all the directories will be created
    """
    dirs = (
        os.path.join(output_base_dir, "real"),
        os.path.join(output_base_dir, "fake"),
        os.path.join(output_base_dir, "temp_frames")
    )

    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs
    
