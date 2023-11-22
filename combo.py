import streamlit as st
import cv2
import numpy as np
from PIL import Image
import face_recognition
import os

# Load the face recognition model
face_1 = face_recognition.load_image_file("E:/Void - Hack/face-recognition/pro-pic.jpeg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]

known_face_encodings = [face_1_encoding]
known_face_names = ["mubeen"]

# Define the directory containing real fingerprint images
real_images_dir = "E:/Void - Hack/face-recognition/Fingerprint - database"
def redirect_button(url: str, text: str= None, color="rgb(19, 23, 67)"):
    st.markdown(
    f"""
    <a href="{url}" target="_self">
        <div style="
            display: inline-block;
            padding: 0.5em 1em;
            color: #FFFFFF;
            background-color: {color};
            border-radius: 3px;
            text-decoration: none;">
            {text}
        </div>
    </a>
    """,
    unsafe_allow_html=True
    )
    
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
    st.title("Biometric Authentication")

    col1, col2 = st.columns(2)

    fingerprint_done = False
    face_done = False

    with col1:
        st.header("Fingerprint Recognition")
        uploaded_file = st.file_uploader("Choose a fingerprint image...", type="bmp")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            sample = cv2.imdecode(file_bytes, 1)

            # Initialize variables to store the best matching result
            best_score = 0
            best_filename = None
            best_image = None
            best_kp1 = None
            best_kp2 = None
            best_mp = None

            # Loop through each real fingerprint image in the directory
            for counter, file in enumerate(os.listdir(real_images_dir)):
                if counter % 10 == 0:
                    st.write("Processing image", counter)

                fingerprint_path = os.path.join(real_images_dir, file)
                fingerprint_img = cv2.imread(fingerprint_path)

                # Check if the image could not be loaded
                if fingerprint_img is None:
                    st.write("Error loading:", fingerprint_path)
                    continue

                # Create a SIFT detector
                sift = cv2.SIFT_create()

                # Detect keypoints and compute descriptors for the sample and real fingerprint images
                keypoints_1, des1 = sift.detectAndCompute(sample, None)
                keypoints_2, des2 = sift.detectAndCompute(fingerprint_img, None)

                # Check if keypoint detection failed for either image
                if keypoints_1 is None or keypoints_2 is None:
                    st.write("Keypoint detection failed for", fingerprint_path)
                    continue

                # Create a FLANN-based matcher for keypoint matching
                matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {})
                matches = matcher.knnMatch(des1, des2, k=2)

                # Filter good matches based on Lowe's ratio test
                match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]

                # Calculate a matching score as the ratio of good matches to total keypoints
                keypoints = min(len(keypoints_1), len(keypoints_2))
                score = len(match_points) / keypoints * 100

                # Update the best matching result if the current score is higher
                if score > best_score:
                    best_score = score
                    best_filename = file
                    best_image = fingerprint_img
                    best_kp1, best_kp2, best_mp = keypoints_1, keypoints_2, match_points

            if(best_score==0):
                st.write("No match found")
            else:
                # Print the best match filename and score
                st.write("Match Found:", best_filename)
                st.write("Match Score:", best_score)

                # Display the best match result if it exists
                if best_mp:
                    result = cv2.drawMatches(sample, best_kp1, best_image, best_kp2, best_mp, None)
                    result = cv2.resize(result, None, fx=5, fy=5)
                    image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(image, caption='Match Found')

                # Optionally, you can save the best match result as an image
                if best_filename:
                    cv2.imwrite("best_match_result.jpg", best_image)

                fingerprint_done = True

    with col2:
        st.header("Face Recognition")
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

                    face_done = True

        # Release the camera
        cap.release()

    if fingerprint_done and face_done:
        # st.markdown('Next{:target="_blank"}', unsafe_allow_html=True)
        redirect_button("http://stackoverflow.com","Next")

if __name__ == "__main__":
    main()
