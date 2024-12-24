import cv2
import numpy as np
import os

def add_badge_with_timestamps(video_file, badge_image, detection_dict):
    """
    Adds a badge to the top-right corner of the video at specific timestamps.

    Args:
        video_file (str): Path to the input video file.
        badge_image (str): Path to the badge image
        detection_dict (dict): Dictionary with timestamps as keys and detection status (1 for deepfakes, 0 for skip) 

    Returns:
        str: Path to the saved output video file.
    """

    # Load the badge image with transparency support
    badge = cv2.imread(badge_image, cv2.IMREAD_UNCHANGED)
    if badge is None:
        raise FileNotFoundError(f"Badge image not found at {badge_image}")

    badge_height, badge_width, badge_channels = badge.shape

    # Ensure the badge has an alpha channel
    if badge_channels != 4:
        alpha_channel = np.ones((badge_height, badge_width), dtype=np.uint8) * 255
        badge = np.dstack((badge, alpha_channel))

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found or cannot be opened at {video_file}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Generate output file name
    output_file = os.path.splitext(video_file)[0] + "_with_badge.mp4"

    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_count}. Ending processing...")
            break

        # Calculate the timestamp
        timestamp = frame_count / fps

        # Check if the badge should be displayed at this timestamp
        if int(timestamp) in detection_dict and detection_dict[int(timestamp)] == 1:
            y_offset = 10
            x_offset = width - badge_width - 10

            # Ensure badge fits within frame dimensions
            if y_offset + badge_height <= height and x_offset + badge_width <= width:
                for c in range(3):  # Iterate over color channels
                    alpha = badge[:, :, 3] / 255.0  # Normalize alpha channel
                    frame[y_offset:y_offset+badge_height, x_offset:x_offset+badge_width, c] = (
                        badge[:, :, c] * alpha +
                        frame[y_offset:y_offset+badge_height, x_offset:x_offset+badge_width, c] * (1 - alpha)
                    )

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Output video saved at: {output_file}")
    return output_file
