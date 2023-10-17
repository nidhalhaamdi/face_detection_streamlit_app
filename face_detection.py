import cv2
import streamlit as st
from PIL import ImageColor

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def app():
    st.set_page_config(layout="wide")

    # Create a sidebar
    st.sidebar.title("Settings")

    # Create a colors bar
    color_hex = st.sidebar.color_picker("Color of the rectangles drawn around the detected faces", '#000000')
    color_rgb = ImageColor.getcolor(color_hex, "RGB")

    # Put slide to adjust the scaleFactor parameter
    scaleFactor = st.sidebar.slider("scaleFactor", 1.0, 2.0, 1.13, 0.01)
    st.sidebar.info("scaleFactor: Parameter specifying how much the image size is reduced at each image scale.")

    # Put slide to adjust the minNeighbors parameter
    minNeighbors = st.sidebar.slider("minNeighbors", 1, 10, 5)
    st.sidebar.info("minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.")
    
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    
    # Add a button to start detecting faces
    if st.button("Detect Faces", type="primary"):
        # Call the detect_faces function
        # Initialize the webcam + Camera Settings
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        FRAME_WINDOW = st.image([])
        
        while True:
            # Read the frames from the webcam
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                st.info("Please turn off the other app that is using the camera and restart app")
                st.stop()
            # Convert the frames to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the faces using the face cascade classifier
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_rgb, 2)
            # Display the frames
            FRAME_WINDOW.image(frame, channels="BGR")
            
            # Make a screenshot and save it
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite('screenshot.jpg',frame)
            
            # Exit the loop when 'ECHAP'(ESC) is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break

if __name__ == "__main__":
    app()