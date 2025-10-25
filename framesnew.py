#better logic based on diff between 2 frmaes
import cv2
import os

def extract_motion_frames(video_path, output_dir="motion_frames", diff_threshold=100000000):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: cannot read video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    saved_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference between current frame and previous frame
        diff = cv2.absdiff(prev_gray, gray)
        diff_sum = diff.sum()  # overall intensity of change

        # If difference is above threshold, save the frame
        if diff_sum > diff_threshold:
            filename = os.path.join(output_dir, f"motion_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Saved motion frame {frame_count} (diff={diff_sum})")

        prev_gray = gray
        frame_count += 1

    cap.release()
    print(f"Total motion frames saved: {saved_count}")

# Example usage
extract_motion_frames("vid1.mp4", diff_threshold=20_000)
