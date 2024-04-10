import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe model
mp_pose = mp.solutions.pose

# To make the video run faster initialize mode_complexity with 0

pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize variables
counter = 0
stage = None
feedback = None
old_feedback = None
frame_count = 0
skip_message = 6
# Thresholds for form evaluation
ANGLE_SHOULDER_HIP_KNEE_THRESHOLD = 160  # Angle Threshold for the angle between shoulder hip knee
ANGLE_HEAD_POSITION_THRESHOLD = 100  # Threshold for head position relative to shoulders
ANGLE_HIP_KNEE_ANKLE_THRESHOLD = 150  # Angle Threshold for the angle between hip knee ankle
ANGLE_DOWN_THRESHOLD = 90  # Angle Threshold for the angle for the down stage
ANGLE_UP_THRESHOLD = 170  # Angle Threshold for the angle for the up stage

# Feedback messages for form correction
FEEDBACK_MESSAGES = {
    "hip": "Adjust Hip Position: Keep your hips aligned with your shoulders.",
    "leg": "Adjust Leg Position: Keep your legs aligned with your hips.",
    "head": "Keep Your Head Aligned: Maintain a neutral head position in line with your spine.",
    "form": "Maintain Proper Form: Focus on maintaining proper alignment and posture."
}

# Colors for feedback indicators
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


# Function to draw text on image with background rectangle
def draw_text(image, text, position, color=(255, 255, 255), bg_color=(0, 0, 0), font_scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x, text_y = position[0], position[1] + text_size[1] // 2

    # Draw background rectangle
    bg_left_top = (text_x - 5, text_y - text_size[1] - 5)
    bg_right_bottom = (text_x + text_size[0] + 5, text_y + 5)
    cv2.rectangle(image, bg_left_top, bg_right_bottom, bg_color, -1)

    # Draw text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)


# Function to draw feedback indicators
def draw_feedback(image, feedback):
    feedback_text = "Form Correct" if feedback is None else FEEDBACK_MESSAGES[feedback]
    feedback_color = GREEN if feedback is None else YELLOW if feedback == "form" else RED
    cv2.circle(image, (60, 60), 10, feedback_color, -1)
    draw_text(image, feedback_text, (80, 60), color=(255, 255, 255), bg_color=feedback_color)


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    angle = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]))
    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle
    return angle

# Function to validate form on each frame
def validate_form(image):
    global counter, stage, feedback, old_feedback, skip_message

    # Perform pose estimation
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # If pose detected
    if results.pose_landmarks:
        # Extract key points for the shoulders, hips, hands, elbows, wrists, head, knees, ankles and ear
        landmarks = results.pose_landmarks.landmark
        points = {
            "left_shoulder": np.array(
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]),
            "right_shoulder": np.array(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]),
            "left_hip": np.array(
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]),
            "right_hip": np.array(
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]),
            "left_wrist": np.array(
                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]),
            "right_wrist": np.array(
                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]),
            "left_elbow": np.array(
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]),
            "right_elbow": np.array(
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]),
            "head": np.array([landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]),
            "left_knee": np.array(
                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]),
            "right_knee": np.array(
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]),
            "left_ankle": np.array(
                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]),
            "right_ankle": np.array(
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]),
            "left_ear": np.array(
                [landmarks[mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR].y]),
            "right_ear": np.array(
                [landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y])
        }
        # Draw main joints
        for point in points.values():
            x, y = int(point[0] * image.shape[1]), int(point[1] * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # Calculate form metrics
        angle_left_shoulder_left_hip_left_knee = calculate_angle(points["left_shoulder"], points["left_hip"],
                                                                 points["left_knee"])
        angle_right_shoulder_right_hip_right_knee = calculate_angle(points["right_shoulder"], points["right_hip"],
                                                                    points["right_knee"])
        angle_left_hip_left_knee_left_ankle = calculate_angle(points["left_hip"], points["left_knee"],
                                                              points["left_ankle"])
        angle_right_hip_right_knee_right_ankle = calculate_angle(points["right_hip"], points["right_knee"],
                                                                 points["right_ankle"])
        angle_left_shoulder_left_hip_left_ear = calculate_angle(points["left_hip"],
                                                                points["left_shoulder"], points["left_ear"])
        angle_right_shoulder_right_hip_right_ear = calculate_angle(points["right_hip"],
                                                                   points["right_shoulder"], points["right_ear"])
        angle_left_shoulder_left_elbow_left_wrist = calculate_angle(points["left_shoulder"],
                                                                    points["left_elbow"], points["left_wrist"])
        angle_right_shoulder_right_elbow_right_wrist = calculate_angle(points["right_shoulder"],
                                                                       points["right_elbow"], points["right_wrist"])
        # Form feedback
        if angle_left_shoulder_left_hip_left_knee < ANGLE_SHOULDER_HIP_KNEE_THRESHOLD or \
                angle_right_shoulder_right_hip_right_knee < ANGLE_SHOULDER_HIP_KNEE_THRESHOLD:
            feedback = "hip"
        elif angle_right_hip_right_knee_right_ankle < ANGLE_HIP_KNEE_ANKLE_THRESHOLD or \
                angle_left_hip_left_knee_left_ankle < ANGLE_HIP_KNEE_ANKLE_THRESHOLD:
            feedback = "leg"
        elif angle_left_shoulder_left_hip_left_ear < ANGLE_HEAD_POSITION_THRESHOLD or \
                angle_right_shoulder_right_hip_right_ear < ANGLE_HEAD_POSITION_THRESHOLD:
            feedback = "head"

        # Check if the person is going up in the push-up
        if angle_left_shoulder_left_elbow_left_wrist <= ANGLE_DOWN_THRESHOLD or \
                angle_right_shoulder_right_elbow_right_wrist <= ANGLE_DOWN_THRESHOLD:
            stage = "down"
        if (angle_left_shoulder_left_elbow_left_wrist >= ANGLE_UP_THRESHOLD or
            angle_right_shoulder_right_elbow_right_wrist >= ANGLE_UP_THRESHOLD) and stage == "down":
            stage = "up"
            counter += 1  # Increment the counter when returning to the starting position

        print(counter)

        # Display the push-up count and form feedback
        draw_text(image, f"Push-ups: {counter}", (20, 10))
        draw_text(image, f"stage: {stage}", (20, 30))

        # Maintain the feedback on the screen for (skip_message) frames

        if frame_count % skip_message == 0:
            old_feedback = feedback
            feedback = None

        draw_feedback(image, old_feedback)

        # Display the image
        cv2.imshow('Push-up Tracker', image)


# Function to capture video feed
def capture_video(video_file):
    global frame_count, feedback, counter, stage
    frame_count = 0
    counter = 0
    feedback = None
    stage = None
    cap = cv2.VideoCapture(video_file)  # To use camera put 0 instead of video_file
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Empty video")
            break

        validate_form(image)

        frame_count += 1

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# List of video file paths
video_files = ["D:\py project/test.mp4",
               "D:\py project\Correct sequence/Copy of push up 83.mp4",
               "D:\py project\Wrong sequence/Copy of push up 150.mp4",
               "D:\py project\Wrong sequence/Copy of push up 155.mp4",
               "D:\py project\Wrong sequence/Copy of push up 110.mp4"]
for video_file in video_files:
    capture_video(video_file)
