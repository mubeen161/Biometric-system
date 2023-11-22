# Import necessary libraries
import os
import cv2
import matplotlib.pyplot as plt

# Define the path to the sample fingerprint image
sample_path = "E:/Void - Hack/face-recognition/test-fingers/Mubeen-test.BMP"
sample = cv2.imread(sample_path)

# Initialize variables to store the best matching result
best_score = 0
best_filename = None
best_image = None
best_kp1 = None
best_kp2 = None
best_mp = None

# Define the directory containing real fingerprint images
real_images_dir = "E:/Void - Hack/face-recognition/Fingerprint - database"

# Loop through each real fingerprint image in the directory
for counter, file in enumerate(os.listdir(real_images_dir)):
    if counter % 10 == 0:
        print("Processing image", counter)

    fingerprint_path = os.path.join(real_images_dir, file)
    fingerprint_img = cv2.imread(fingerprint_path)

    # Check if the image could not be loaded
    if fingerprint_img is None:
        print("Error loading:", fingerprint_path)
        continue

    # Create a SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the sample and real fingerprint images
    keypoints_1, des1 = sift.detectAndCompute(sample, None)
    keypoints_2, des2 = sift.detectAndCompute(fingerprint_img, None)

    # Check if keypoint detection failed for either image
    if keypoints_1 is None or keypoints_2 is None:
        print("Keypoint detection failed for", fingerprint_path)
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
    print("No match found")

else:
# Print the best match filename and score
    print("Match Found:", best_filename)
    print("Match Score:", best_score)

# Display the best match result if it exists
if best_mp:
    result = cv2.drawMatches(sample, best_kp1, best_image, best_kp2, best_mp, None)
    result = cv2.resize(result, None, fx=5, fy=5)
    image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title("Match Found")
    plt.axis("off")
    plt.show()

# Optionally, you can save the best match result as an image
if best_filename:
    cv2.imwrite("best_match_result.jpg", best_image)