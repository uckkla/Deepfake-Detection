import { useState, useRef, useEffect } from "react";
import "./FileUploader.css"

const FileUploader = () => {
    const [files, setFiles] = useState(null);
    const inputRef = useRef();

    const handleDragOver = (event) => {
        event.preventDefault();
    };

    const handleDrop = (event) => {
        event.preventDefault();
        setFiles(event.dataTransfer.files);
    };

    const handleUpload = () => {

    };

    const handleFileSelect = (event) => {
        const selectedFiles = Array.from(event.target.files);
        setFiles((prevFiles) => [...(prevFiles || []), ...selectedFiles]);

        // Reset value so new files are always accepted
        event.target.value = "";
    };

    const handleRemoveFile = (index) => {
        const fileArr = Array.from(files);
        fileArr.splice(index, 1);

        if (fileArr.length === 0){
            setFiles(null);
        }
        else{
            setFiles(fileArr);
        }
    };

    const formatFileSize = (size) => {
        if (size < 1024) {
            return `${size} B`;
        } else if (size < 1024 * 1024) {
            return `${(size / 1024).toFixed(2)} KB`;
        } else {
            return `${(size / (1024 * 1024)).toFixed(2)} MB`;
        }
    };

    /*
    useEffect(() => {
        console.log("Updated files:", files);
    }, [files]);
    */

    return (
    <div className="uploader">
        <input type="file" multiple onChange={handleFileSelect} hidden ref={inputRef}
    />
        {files ? (
            <div className="uploads">
            {Array.from(files).map((file, index) => (
                <div key={index} className="file-item">
                    <div className="file-info">
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">{formatFileSize(file.size)}</span>
                    </div>
                    <button onClick={() => handleRemoveFile(index)}>Remove</button>
                </div>
            ))}
                <div className="actions">
                    <button onClick={() => inputRef.current.click()}>Add more Files</button>
                    <button onClick={handleUpload}>Upload</button>
                </div>
            </div>
        ) : (
            <div className="dropzone" onDragOver={handleDragOver} onDrop={handleDrop}>
                <h1>Drag and Drop Files</h1>
                <h1>Or</h1>
                <button onClick={() => inputRef.current.click()}>Select Files</button>
            </div>
       )} 
    </div>
    );
};

export default FileUploader;