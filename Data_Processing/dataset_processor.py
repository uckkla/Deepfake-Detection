import json
import os
import shutil
from Data_Processing.face_extraction import extract_faces
from Data_Processing.frame_extraction import extract_frames
import random

def process_dfdc_videos(video_dir, output_base_dir, target_fps=5):
    """
    Processing videos provided by DFDC (Deepfake Detection Challenge), Meta
    Source: https://www.kaggle.com/c/deepfake-detection-challenge/data
    Args:
        video_path: Input video file path
        output_base_dir: Directory containing dataset
        target_fps: Target frames per second
    """
    processed = 0

    # Metadata contains data of all videos - if they are fake or real
    with open(os.path.join(video_dir, "metadata.json")) as metadata_json:
        metadata = json.load(metadata_json)

    # Separate REAL and FAKE video files
    real_videos = [fname for fname, meta in metadata.items() if meta["label"] == "REAL"]
    fake_videos = [fname for fname, meta in metadata.items() if meta["label"] == "FAKE"]
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos.")

    # Downsample the larger set to match the smaller one
    min_len = min(len(real_videos), len(fake_videos))
    real_videos = random.sample(real_videos, min_len)
    fake_videos = random.sample(fake_videos, min_len)

    balanced_videos = real_videos + fake_videos
    random.shuffle(balanced_videos)

    real_dir, fake_dir, temp_dir = create_required_directories(output_base_dir)
    
    for filename in balanced_videos:
        label = metadata[filename]["label"]
        video_path = os.path.join(video_dir, filename)

        if not os.path.exists(video_path):
            print(f"Warning: Missing video file {filename}")
            continue
        
        output_dir = real_dir if label == "REAL" else fake_dir

        process_video(
            video_path=video_path,
            output_dir=output_dir,
            temp_dir=temp_dir,
            target_fps=target_fps
        )

        processed += 1
        print(f"{processed}/{len(balanced_videos)} videos processed.")

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

    celeb_fake_dir = os.path.join(video_dir, "Celeb-synthesis")
    fake_videos = [os.path.join(celeb_fake_dir, f) for f in os.listdir(celeb_fake_dir) if os.path.isfile(os.path.join(celeb_fake_dir, f))]

    real_videos = []
    for real_subdir in ["Celeb-real", "YouTube-real"]:
        real_input_dir = os.path.join(video_dir, real_subdir)
        real_videos += [os.path.join(real_input_dir, f) for f in os.listdir(real_input_dir) if os.path.isfile(os.path.join(real_input_dir, f))]
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos.")

    min_len = min(len(real_videos), len(fake_videos))
    real_videos = random.sample(real_videos, min_len)
    fake_videos = random.sample(fake_videos, min_len)

    processed = 0
    total = len(real_videos) + len(fake_videos)

    for video_type in [real_videos, fake_videos]:
        output_dir = real_dir if video_type is real_videos else fake_dir
        for video_path in video_type:
            process_video(
                video_path=video_path,
                output_dir=output_dir,
                temp_dir=temp_dir,
                target_fps=target_fps
            )

            processed += 1
            print(f"{processed}/{total} videos processed.")

def process_faceforensics_videos(video_dir, output_base_dir, target_fps=5):
    """
    Processing videos provided by FaceForensics
    https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/data
     Args:
        video_path: Input video file path
        output_base_dir: Directory containing dataset
        target_fps: Target frames per second
    """
    real_dir, fake_dir, temp_dir = create_required_directories(output_base_dir)

    real_input_dir = os.path.join(video_dir, "DFD_original sequences")
    fake_input_dir = os.path.join(video_dir, "DFD_manipulated_sequences")

    real_videos = [os.path.join(real_input_dir, f) for f in os.listdir(real_input_dir) if os.path.isfile(os.path.join(real_input_dir, f))]
    fake_videos = [os.path.join(fake_input_dir, f) for f in os.listdir(fake_input_dir) if os.path.isfile(os.path.join(fake_input_dir, f))]

    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos.")

    min_len = min(len(real_videos), len(fake_videos))
    real_videos = random.sample(real_videos, min_len)
    fake_videos = random.sample(fake_videos, min_len)

    processed = 0
    total = len(real_videos) + len(fake_videos)

    for video_type in [real_videos, fake_videos]:
        output_dir = real_dir if video_type is real_videos else fake_dir
        for video_path in video_type:
            process_video(
                video_path=video_path,
                output_dir=output_dir,
                temp_dir=temp_dir,
                target_fps=target_fps
            )

            processed += 1
            print(f"{processed}/{total} videos processed.")

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
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        extract_frames(
            video_path=video_path,
            output_folder=temp_dir,
            fps=target_fps
        )

        video_id = os.path.splitext(os.path.basename(video_path))[0]
        extract_faces(
            frame_dir=temp_dir,
            output_dir=output_dir,
            filename_prefix=video_id,
            single_image=False
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

def process_user_video(video_path, output_dir, target_fps=5):
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    process_video(
        video_path=video_path,
        output_dir=output_dir,
        temp_dir=temp_dir,
        target_fps=target_fps
    )

def process_user_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    #image_dir = os.path.dirname(image_path)
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    extract_faces(
        frame_dir=image_path,
        output_dir=output_dir,
        filename_prefix=image_id,
        single_image=True
    )

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
    

if __name__ == "__main__":
    #video_path = r"D:\Final Year Project\Deepfake Resources\Celeb-DF-v2"
    #output_dir = r"D:\Final Year Project\Processed Datasets\celeb-df-v2"
    #video_path = r"D:\Final Year Project\Deepfake Resources\dfdc_train_part_10\dfdc_train_part_10"
    #output_dir = r"D:\Final Year Project\Processed Datasets\dfdc\dfdc10"
    video_path = r"D:\Final Year Project\Deepfake Resources\faceforensics"
    output_dir = r"D:\Final Year Project\Processed Datasets\faceforensics"
    target_fps = 2
    #process_celeb_df_videos(video_path, output_dir, target_fps)
    #process_dfdc_videos(video_path, output_dir, target_fps)
    process_faceforensics_videos(video_path, output_dir, target_fps)