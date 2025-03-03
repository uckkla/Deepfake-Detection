from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "mp3"}

app = Flask(__name__)
CORS(app, origins="http://localhost:5173")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def AllowedFile(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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


if __name__ == "__main__":
    # Create upload folder if does not exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)