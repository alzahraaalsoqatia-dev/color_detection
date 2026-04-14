import cv2
import numpy as np

def get_color_name(h, s, v):
    if v < 80:
        return "Black"
    if s < 50:
        if v > 170:
            return "White"
        return "Gray"
    if h < 10 or h >= 165:
        return "Red"
    elif h < 22:
        return "Orange"
    elif h < 38:
        return "Yellow"
    elif h < 85:
        return "Green"
    elif h < 100:
        return "Cyan"
    elif h < 135:
        return "Blue"
    elif h < 155:
        return "Purple"
    else:
        return "Pink"

BOX_COLORS = {
    "Red":    (0, 0, 220),
    "Orange": (0, 130, 255),
    "Yellow": (0, 220, 220),
    "Green":  (0, 180, 0),
    "Cyan":   (200, 200, 0),
    "Blue":   (220, 80, 0),
    "Purple": (160, 0, 160),
    "Pink":   (180, 100, 220),
    "White":  (200, 200, 200),
    "Gray":   (120, 120, 120),
    "Black":  (40, 40, 40),
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Lower varThreshold = more sensitive, higher = less noise
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=100, detectShadows=False
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    frame_area = h_frame * w_frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fg_mask = back_sub.apply(frame)

    # Stronger cleanup to remove fragmented blobs
    kernel_open  = np.ones((15, 15), np.uint8)
    kernel_close = np.ones((25, 25), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_open)   # kill speckles
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)  # fill holes
    fg_mask = cv2.dilate(fg_mask, np.ones((11,11), np.uint8), iterations=1)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Only keep the LARGEST contour — avoids detecting background noise blobs
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Must be between 3% and 85% of frame
        if frame_area * 0.03 < area < frame_area * 0.85:
            x, y, bw, bh = cv2.boundingRect(largest)

            # Sample pixels inside the contour
            contour_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
            cv2.drawContours(contour_mask, [largest], -1, 255, -1)
            pixels = hsv[contour_mask > 0]

            if len(pixels) >= 50:
                mean_h = int(np.mean(pixels[:, 0]))
                mean_s = int(np.mean(pixels[:, 1]))
                mean_v = int(np.mean(pixels[:, 2]))

                print(f"HSV -> H:{mean_h} S:{mean_s} V:{mean_v}")

                color_name = get_color_name(mean_h, mean_s, mean_v)
                box_color  = BOX_COLORS.get(color_name, (0, 255, 0))

                cv2.rectangle(frame, (x, y), (x+bw, y+bh), box_color, 3)
                (tw, th), _ = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x, y-th-12), (x+tw+8, y), box_color, -1)
                cv2.putText(frame, color_name, (x+4, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Color Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()