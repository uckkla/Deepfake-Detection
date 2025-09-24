#from Model.train_model import DeepfakeDetector
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import argparse
from Data_Processing.dataset_processor import process_user_video, process_user_image

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def AllowedFile(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Temporary fix - need to link implementation to method in train_model.py
def preprocess_image(path, input_shape=(224, 224)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, input_shape)
    image = image / 255.0  # Normalisation
    #print(tf.reduce_min(image), tf.reduce_max(image), tf.reduce_mean(image))

    return image

def CreateApp(model_path):
    app = Flask(__name__)
    CORS(app, origins="http://localhost:5173")
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    os.makedirs(os.path.join(CURRENT_DIRECTORY, UPLOAD_FOLDER), exist_ok=True)

    model = load_model(model_path)

    @app.route("/uploads", methods=["POST"])
    def UploadFiles():
        if "files" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        uploadedFiles = request.files.getlist("files")
        # Used for telling user which files saved/failed
        savedFiles, failedFiles = [], []

        for file in uploadedFiles:
            if file and AllowedFile(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(CURRENT_DIRECTORY, app.config["UPLOAD_FOLDER"], filename)
                os.makedirs(os.path.dirname(upload_path), exist_ok=True)
                file.save(upload_path)
                savedFiles.append(filename)
            else:
                failedFiles.append(file.filename)

        if not savedFiles:
            return jsonify({"error": "No valid files uploaded", "disallowed_files": failedFiles}), 400
        
        response = {"uploaded": savedFiles}

        if failedFiles:
            response["disallowed_files"] = failedFiles

        # Returns 200 if at least 1 file is successfully uploaded
        return jsonify(response), 200

    @app.route("/analyse", methods=["POST"])
    def AnalyseFile():
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"error": "Filename not provided"}), 400
        
        filepath = os.path.join(CURRENT_DIRECTORY, app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        
        file_ext = filename.rsplit(".", 1)[1].lower()
        face_dir = os.path.join(CURRENT_DIRECTORY, "temp_faces", filename.rsplit(".", 1)[0])
        os.makedirs(face_dir, exist_ok=True)

        try:
            if file_ext in ALLOWED_IMAGE_EXTENSIONS:
                process_user_image(filepath, face_dir)
            elif file_ext in ALLOWED_VIDEO_EXTENSIONS:
                process_user_video(filepath, face_dir)
            else:
                return jsonify({"error": "Unsupported file type"}), 400
            
            face_paths = [
                os.path.join(face_dir, f)
                for f in os.listdir(face_dir)
                if f.lower().endswith(".jpg")
            ]

            if not face_paths:
                return jsonify({"error": "No faces detected"}), 400
            
            predictions = []
            for face_path in face_paths:
                try:
                    img = preprocess_image(path=face_path)
                    img = tf.expand_dims(img, axis=0)  # Add batch dimension
                    pred = model.predict(img)[0][0]
                    print(f"Prediction for {face_path}: {pred}")
                    predictions.append(pred)      
                except Exception as e:
                    print(f"Error processing {face_path}: {e}")

            if not predictions:
                return jsonify({"error": "Face preprocessing or prediction failed"}), 500

            predictions = np.array(predictions)
            
            # Interquartile Range filtering
            q1 = np.percentile(predictions, 25)
            q3 = np.percentile(predictions, 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            filtered = predictions[(predictions >= lower_bound) & (predictions <= upper_bound)]

            prediction = float(np.mean(filtered))

            return jsonify({
                "filename": filename,
                "confidence": float(prediction),
                "label": "Fake" if prediction > 0.5 else "Real"
            })

        finally:
            import shutil
            shutil.rmtree(face_dir, ignore_errors=True)

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deepfake detection API")
    parser.add_argument(
        "--model_path", type=str, default=os.path.join("Model", "best_model.keras"),
        help="Path to the trained Keras model file (include file name)"
    )
    args = parser.parse_args()
    app = CreateApp(args.model_path)
    app.run(debug=True)