from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
#from Model.train_model import DeepfakeDetector
from Data_Processing.dataset_processor import process_user_video, process_user_image
import os
import tensorflow as tf

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = os.path.join("Model", "final_model", "best_model.keras")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

app = Flask(__name__)
CORS(app, origins="http://localhost:5173")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model(MODEL_PATH)

def AllowedFile(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Temporary fix - need to link implementation to methodin train_model.py
def preprocess_image(path, input_shape=(224, 224)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, input_shape)
    image = image / 255.0  # Normalisation

    return image

@app.route("/uploads", methods=["POST"])
def UploadFiles():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    uploadedFiles = request.files.getlist("files")
    # Used for telling user which files saved/failed
    savedFiles = []
    failedFiles = []

    for file in uploadedFiles:
        if file and AllowedFile(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            savedFiles.append(filename)
        else:
            failedFiles.append(file.filename)
    
    response = {"uploaded": savedFiles}

    if failedFiles:
        response["disallowed_files"] = failedFiles

    # Returns 200 if at least 1 file is successfully uploaded
    return jsonify(response), 200 if savedFiles else 400

@app.route("/analyse", methods=["POST"])
def AnalyseFile():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Filename not provided"}), 400
    
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    file_ext = filename.rsplit(".", 1)[1].lower()

    # Directory to save extracted faces temporarily
    face_dir = "user_faces"
    os.makedirs("user_faces", exist_ok=True)

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
                predictions.append(pred)      
            except Exception as e:
                print(f"Error processing {face_path}: {e}")

        if not predictions:
            return jsonify({"error": "Face preprocessing or prediction failed"}), 500

        prediction = sum(predictions) / len(predictions)

        return jsonify({
            "filename": filename,
            "confidence": float(prediction),
            "label": "Fake" if prediction > 0.5 else "Real"
        })

    finally:
        # Optional: cleanup extracted face images after prediction
        import shutil
        shutil.rmtree(face_dir, ignore_errors=True)

if __name__ == "__main__":
    # Create upload folder if does not exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)