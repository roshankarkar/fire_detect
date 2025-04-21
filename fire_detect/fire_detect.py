import cv2
import numpy as np
import pygame  # For playing sound

# Initialize pygame mixer
pygame.mixer.init()

# Function to play sound
def play_sound(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# Function to detect fire based on motion, color, and flickering behavior
def detect_fire_by_motion(frame, prev_frame):
    if prev_frame is None:
        return False, None

    # Convert the current and previous frames to grayscale
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current and previous frames
    diff = cv2.absdiff(gray_prev_frame, gray_frame)

    # Apply a threshold to the difference to identify significant motion
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Filter out small regions that are unlikely to be fire
        if area > 500:  # Increased minimum area to reduce false positives
            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the region of interest (ROI) from the original frame
            roi = frame[y:y+h, x:x+w]

            # Convert the ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define the range of fire-like colors in HSV
            lower_fire = np.array([0, 50, 200])  # Stricter lower bound
            upper_fire = np.array([35, 255, 255])  # Stricter upper bound

            # Create a mask for fire-like colors
            mask = cv2.inRange(hsv_roi, lower_fire, upper_fire)

            # Calculate the percentage of fire-like pixels in the ROI
            fire_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            fire_ratio = fire_pixels / total_pixels

            # Check for flickering behavior (optional: track over time)
            if fire_ratio > 0.4:  # Increased threshold for fire-like color ratio
                return True, (x, y, w, h)

    return False, None

# Start video capture
cap = cv2.VideoCapture(0)

fire_detected = False  # To avoid playing the sound repeatedly
prev_frame = None  # To store the previous frame for motion detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Detect fire based on motion and color
    is_fire_detected, fire_region = detect_fire_by_motion(frame, prev_frame)
    if is_fire_detected:
        x, y, w, h = fire_region
        # Draw a rectangle only around the detected fire region
        cv2.putText(frame, "FIRE FIRE FIRE! RUN FROM HERE!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw a rectangle around the fire region
        if not fire_detected:
            fire_detected = True
            print("FIRE FIRE FIRE! RUN FROM HERE!")
            play_sound(r"D:\github\fire_detect_py\alert.wav")  # Replace with the correct path to your MP3 file

        # Display the frame with fire and stop updating
        cv2.imshow("Fire Detection", frame)
        print("Fire detected. Pausing frame updates.")
        cv2.waitKey(0)  # Wait indefinitely until the user presses a key
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Update the previous frame
    prev_frame = frame.copy()

    # Display the frame
    cv2.imshow("Fire Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
