# Deepfake Detector
## Overview
A deep learning-based web application for detecting deepfakes in images and videos using facial classification.
The system uses an EfficientNetB0 model with a custom head and supports video/image uploads through a web interface, with server-side processing and model inference.
<img width="1904" height="949" alt="web_interface" src="https://github.com/user-attachments/assets/545a135e-6940-490e-9004-bb954508ebe2" />


## Features
- Detects whether a media file (image or video) is real or fake using a confidence score
- Users can upload media through a web interface, using React as the frontend
- Backend uses Flask, TensorFlow, and MTCNN for face detection
- EfficientNetB0-based model trained on Celeb-DF v2, DFDC, and FaceForensics++ datasets
- Includes custom data preprocessing and balanced dataset creation
- Real-time face extraction, classification, and response on user upload

## Datasets
This project uses publicly available deepfake datasets for training and evaluation:
- **DFDC (Deepfake Detection Challenge)**  
  [Kaggle Dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge) · [Meta AI Dataset](https://ai.meta.com/datasets/dfdc/)  
  A large-scale dataset of manipulated and real videos released as part of Facebook’s deepfake detection challenge.
  
- **FaceForensics++**  
  [Project Page (Request Dataset)](https://github.com/ondyari/FaceForensics) · [Paper (Rossler et al., 2019)](https://arxiv.org/pdf/1901.08971)  
  A benchmark dataset containing various manipulation methods applied to real-world videos.

- **Celeb-DF v2**  
  [Kaggle Dataset](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) · [Project Page](https://github.com/yuezunli/celeb-deepfakeforensics)  
  A high-quality deepfake dataset addressing limitations of earlier collections.

> [!IMPORTANT]
> Make sure that the folder structure is not altered when using the public datasets.

## Installation & Usage
### Clone the repository using git
```
git clone https://github.com/uckkla/Deepfake-Detection.git
```
### Setup Virtual Environment
```
cd Deepfake-Detection
python -m venv .venv
.venv\Scripts\Activate
```

### Install Python Packages
```
cd Deepfake-Detection
pip install -r requirements.txt
```

### Prepare Dataset
```
# Example for DFDC dataset
python -m Data_Processing.dataset_processor --video_path path/to/dfdc --outputdir path/to/output --target_fps 2 --dataset dfdc
```
> [!NOTE]
> --dataset options: dfdc, celebdf, and faceforensics.
> --target_fps defaults to 2 if not specified.

### Train the model
```
python -m Model.train_model --data_dir path/to/data --model_dir path/to/model --epochs 30 --batch_size 32
```
> [!NOTE]
> You can adjust --epochs and --batch_size depending on your hardware and dataset size.
> More epochs improve performance but increase training time.

> [!TIP]
> Training may involve multiple stages (e.g., freezing base layers, fine-tuning later), so at least 30 epochs is recommended.

### Run Frontend and Backend
```
# Run on Terminal 1:
python -m Server.main
```
```
# Run on Terminal 2:
cd Deepfake-Detection\Website
npm install
npm run dev
```

## Using the Interface
Once both the backend and frontend are running, open the web application in your browser: [http://localhost:5173](http://localhost:5173)

### Uploading Files
- Click the **Select Files** button and choose an image (`.jpg`, `.png`) or video (`.mp4`, `.avi`, `.mov`, `.mkv`).
- Alternatively, drag and drop media files directly onto the interface.
- Click the **Upload** button to send the selected files to the server.

### Analysing Files
- Once uploaded, each file will appear in the list with an **Analyse** button.  
- Click **Analyse** to process the file:
  - For images: the system extracts the face and classifies it as **Real** or **Fake**.  
  - For videos: multiple frames are sampled, faces are extracted, and predictions are aggregated for a more reliable result.

### Results
- After analysis, results will be displayed on the interface, including:
  - **Filename** of the uploaded media  
  - **Confidence score** (between 0.0 and 1.0)  
  - **Predicted label**: **Real** (≤ 0.5) or **Fake** (> 0.5)

> [!TIP]  
> For videos, the system applies interquartile filtering to remove outlier predictions before averaging. The longer the video, the more accurate the result will be.
