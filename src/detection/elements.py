"""Detection helpers: screen position, bbox ratio, and head pose estimation."""

import cv2
import mediapipe as mp
import numpy as np

GRID_LABELS = {
    0: "왼쪽 위",
    1: "위",
    2: "오른쪽 위",
    3: "왼쪽",
    4: "가운데",
    5: "오른쪽",
    6: "왼쪽 아래",
    7: "아래",
    8: "오른쪽 아래",
}


def detect(xmin, ymin, xmax, ymax, frame):
    """Return bbox area percentage and 3x3 grid location in Korean."""
    bbox_area = (xmax - xmin) * (ymax - ymin)
    pct = round(100 * bbox_area / (frame.shape[0] * frame.shape[1]))

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    col = int(x_center / frame.shape[1] * 3)
    row = int(y_center / frame.shape[0] * 3)
    loc = GRID_LABELS[row * 3 + col]
    return pct, loc


face_detect = detect  # backward-compatible alias


def head_pose(image, face_mesh):
    """Estimate head direction using MediaPipe Face Mesh and OpenCV solvePnP."""
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    text = "No face detected"
    if not results.multi_face_landmarks:
        return text

    img_h, img_w = image.shape[:2]
    for face_landmarks in results.multi_face_landmarks:
        face_3d = []
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in (33, 263, 1, 61, 291, 199):
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array(
            [
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1],
            ]
        )
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat = cv2.Rodrigues(rot_vec)[0]
        angles = cv2.RQDecomp3x3(rmat)[0]

        x_angle = angles[0] * 360
        y_angle = angles[1] * 360

        if y_angle < -5:
            text_h = f"looking left {round(y_angle, 1)}"
        elif y_angle > 5:
            text_h = f"looking right {round(y_angle, 1)}"
        else:
            text_h = "looking forward"

        if x_angle < -3.5:
            text_v = f"looking down {round(x_angle, 1)}"
        elif x_angle > 4:
            text_v = f"looking up {round(x_angle, 1)}"
        else:
            text_v = ""

        text = f"{text_h} {text_v}".strip()

    return text


# backward-compatible alias
head_Pose = head_pose


def box_to_pct(boxes, width, height):
    """Convert bounding boxes to area percentage of the frame."""
    pct = []
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        pct.append((xmax - xmin) * (ymax - ymin) / (width * height) * 100)
    return pct
