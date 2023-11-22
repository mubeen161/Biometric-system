import streamlit as st
import cv2
import numpy as np
from PIL import Image
import face_recognition

# Load the face recognition model
face_1 = face_recognition.load_image_file("E:/Void - Hack/face-recognition/pro-pic.jpeg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]

known_face_encodings = [face_1_encoding]
known_face_names = ["mubeen"]

# Function to perform face recognition
# Function to perform face recognition
# Function to perform face recognition
def recognize_faces(image_array):
    # Save the image to a temporary file
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    # Load the image using face_recognition, which handles color space correctly
    loaded_image = face_recognition.load_image_file(temp_image_path)
    face_locations = face_recognition.face_locations(loaded_image)
    face_encodings = face_recognition.face_encodings(loaded_image, face_locations)

    result_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV processing

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(result_image, name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)




def main():
    st.title("Face Recognition App")

    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to open the camera.")
    else:
        st.write("Adjust your position.")

        capture_button = st.button("Capture Image")

        if capture_button:
            st.write("Image captured! Processing...")

            # Capture image from webcam
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # st.image(frame, caption="Captured Image", use_column_width=True)

                # Perform face recognition
                result_image = recognize_faces(frame)

                # Display the result
                st.image(result_image, caption="Resulted Image", use_column_width=True)
                retake_button = st.button("Retake Image")
                if retake_button:
                    st.write("Adjust your position.")
                else:
                    # Release the camera
                    cap.release()

    # Release the camera
    cap.release()

if __name__ == "__main__":
    main()
