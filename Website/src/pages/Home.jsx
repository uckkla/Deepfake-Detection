import FileUploader from "../components/FileUploader.jsx"

const Home = () => {
    return (
        <>
            <div className="infoSection">
                <h2>Deepfake Detector</h2>
                <p>This application allows you to upload videos, which are then analysed for
                    any deepfakes. A confidence score is outputted, showing the probability the
                    video is deepfaked. Below you can choose to drag the file into the dropzone,
                    or select the file from your desktop.
                </p>
            </div>
            <FileUploader/>
        </>
    );
};

export default Home;