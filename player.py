import cv2
import numpy as np
import time

FPS = 30

scenes = [
    "media/videos/presentation/480p15/SchrodingerEquation.mp4",
    "media/videos/presentation/480p15/SchrodingerEquation.mp4",
]
# scenes = [
#     "media/videos/presentation/1080p15/SchrodingerEquation.mp4",
#     "media/videos/presentation/1080p15/SchrodingerEquation.mp4",
# ]

i = 0
t0 = time.time()
while i < len(scenes):
    paused = False
    scene = cv2.VideoCapture(scenes[i])
    prev, prev_stop_frame = 0, 0
    print("showing scene", i)

    frame_count = 0
    while scene.isOpened():
        if paused:
            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely
            if key == ord(" "):
                paused = not paused
                prev_stop_frame = frame_count
                continue
            elif key == 8:  # 8 == ascii for backspace.
                i = max(-1, i - 2)
                break
            elif key == ord("q"):
                i = len(scenes)
                break

        ret, frame = scene.read()
        if ret:
            if (
                not paused
                and frame_count - prev_stop_frame > FPS / 2
                and np.all(frame == prev)
            ):
                paused = True
                prev_stop_frame = frame_count
            prev = frame

            delta = time.time() - t0
            if cv2.waitKey(max(1, int(1000 * (1 / FPS - delta)))) & 0xFF == ord(" "):
                paused = not paused

            cv2.imshow("Frame", frame)
            t0 = time.time()

        else:
            break
        frame_count += 1

    i += 1
    scene.release()

cv2.destroyAllWindows()
