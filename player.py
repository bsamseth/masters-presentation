import cv2
import numpy as np
import time

RESOLUTION = "480p15"
FPS = 30
FPS_SOURCE = int(RESOLUTION.split('p')[-1])

scenes = [
    # "TitleScreen",
    # "Outline",
    "Newton",
    "SchrodingerEquation",
]

scenes = [
    ("media/videos/presentation/{}/" + s + ".mp4").format(RESOLUTION)
    for s in scenes
]


# Compile all scenes:
frames = []
for scene in scenes:
    vid = cv2.VideoCapture(scene)
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frames.append(frame)
    vid.release()

print(len(frames))

i = 0
paused = False
direction = 1
prev, prev_stop_frame, t0 = 0, 0, 0
while True:
    frame = frames[i]
    cv2.imshow("Frame", frame)

    delta = time.time() - t0
    wait_time = max(1, int(1000 * (1 / FPS - delta))) * int(not paused)
    key = cv2.waitKey(wait_time) & 0xff

    if key == ord(' '):
        paused = not paused
        prev_stop_frame = i
    elif key == 81:
        i = max(0, i - FPS // 2)
    elif key == 85:
        i = min(len(frames - 1), i + FPS // 2)
    elif key == ord('q'):
        break
    elif key == ord('h'):
        direction = -1
    elif key == ord('l'):
        direction = 1
    elif not paused and abs(i - prev_stop_frame) > FPS_SOURCE / 2 and np.all(frame == prev):
        paused = True
        prev_stop_frame = i

    prev = frame
    t0 = time.time()
    i = max(0, min(len(frames)-1, i + direction))


cv2.destroyAllWindows()


