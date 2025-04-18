import cv2
import numpy as np
import pygame  # For playing sound

# Initialize pygame mixer
pygame.mixer.init()

# Function to play sound
def play_sound(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# Function to detect fire based on motion
def detect_fire_by_motion(frame, prev_frame):
    if prev_frame is None:
        return False, None

    # Convert the current and previous frames to grayscale
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current and previous frames
    diff = cv2.absdiff(gray_prev_frame, gray_frame)

    # Apply a threshold to the difference to identify significant motion
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)  # Lowered threshold for more sensitivity

    # Perform morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Smaller kernel for distant fire
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Filter out small regions that are unlikely to be fire
        if area > 200:  # Lowered area threshold for distant fire
            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the region has a fire-like shape (e.g., tall and narrow)
            aspect_ratio = h / float(w)
            if aspect_ratio > 1.1:  # Lowered aspect ratio for distant fire
                # Analyze the intensity of motion in the region
                roi_diff = diff[y:y+h, x:x+w]
                motion_intensity = np.sum(roi_diff) / 255  # Count the number of white pixels

                # Print debugging information
                print(f"Area: {area}, Aspect Ratio: {aspect_ratio}, Motion Intensity: {motion_intensity}")

                # Fire motion typically has a high intensity and irregular shape
                if motion_intensity > 100:  # Lowered motion intensity threshold for distant fire
                    # Check for irregularity in the contour shape (fire is not perfectly rectangular)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if 0.4 < circularity < 1.6:  # Adjusted circularity range for distant fire
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

    # Detect fire based on motion
    is_fire_detected, fire_region = detect_fire_by_motion(frame, prev_frame)
    if is_fire_detected:
        x, y, w, h = fire_region
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