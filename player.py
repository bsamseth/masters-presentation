import cv2
import numpy as np

# np.set_printoptions(threshold=np.inf)
import time
from collections import deque
import threading

# Low Quality
# PAUSE_INDICATOR = (-1, 0)
# RESOLUTION = "480p15"
# FPS = 15

# Production quality
PAUSE_INDICATOR = (-1, 0)
RESOLUTION = "1440p60"
FPS = 60

cv2.namedWindow("Frame", 0);
cv2.resizeWindow("Frame", *[int(d) for d in RESOLUTION.split('p')])
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

scenes = [
    "TitleScreen",
    "Outline",
    "Newton",
    "SchrodingerEquation",
    "HarmonicOscillator",
    "VMC",
    "WhatToGuess",
    "PsiDesign",
    "NewIdea",
    "NetworkDisplay",
    "QDResults",
    "HeliumResults",
    "Conclusions",
    "FutureProspects",
    "ThankYou",
]
scenes = [
    ("media/videos/presentation/{}/" + s + ".mp4").format(RESOLUTION) for s in scenes
]


class Manager:
    def __init__(self, scenes):
        self.scenes = scenes
        self.active_scene = cv2.VideoCapture(self.scenes[0])
        self.forward = deque()
        self.current_scene = 0
        self.frame = 0
        self.lock = threading.Lock()
        self.last_scene = len(scenes) - 1
        self.keep_running = True

    def stop(self):
        self.lock.acquire()
        self.keep_running = False
        self.active_scene.release()
        self.lock.release()

    def run(self):

        while self.keep_running:
            self.lock.acquire()
            if len(self.forward) >= FPS:
                self.lock.release()
                time.sleep(1 / FPS / 5)
                continue

            if not self.active_scene.isOpened():
                self.active_scene.release()
                self.current_scene = min(self.last_scene, self.current_scene + 1)
                self.active_scene = cv2.VideoCapture(self.scenes[self.current_scene])

            if self.active_scene.isOpened():
                ret, frame = self.active_scene.read()
                if ret:
                    self.forward.append(frame)
                else:
                    self.active_scene.release()
                    self.current_scene = min(self.last_scene, self.current_scene + 1)
                    self.active_scene = cv2.VideoCapture(
                        self.scenes[self.current_scene]
                    )

            self.lock.release()

    def next_frame(self):
        self.lock.acquire()
        frame = self.forward.popleft() if self.forward else None
        self.lock.release()
        return frame

    def play(self):

        paused = False
        indicator_present = False
        t0 = 0
        while True:
            t1 = time.time()
            wait_time = max(1, int(900 * (1 / FPS - (t1 - t0)))) * int(not paused)
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord("q"):
                self.stop()
                break
            elif key == ord(" "):
                paused = not paused
                continue
            elif key == 83:
                for _ in range(FPS // 2):
                    frame = self.next_frame()
                if frame is not None:
                    cv2.imshow("Frame", frame)
            elif key == 81:
                self.active_scene.release()
                self.current_scene = max(0, self.current_scene - 1)
                self.active_scene = cv2.VideoCapture(self.scenes[self.current_scene])
                paused = False
            elif key == ord('n'):
                self.active_scene.release()
                self.current_scene = min(self.last_scene, self.current_scene + 1)
                self.active_scene = cv2.VideoCapture(self.scenes[self.current_scene])
                paused = False
            elif key != 0xFF:
                print("\rUnknown key pressed:", key)

            print(f"{1 / (t1 - t0):.2f}", end="\r")
            if not paused:
                frame = self.next_frame()
                if frame is not None:
                    ind_pres = (
                        frame[PAUSE_INDICATOR][0] == 0
                        and frame[PAUSE_INDICATOR][1] == 0
                        and frame[PAUSE_INDICATOR][2] >= 224
                    )
                    if indicator_present and not ind_pres:
                        paused = True

                    indicator_present = ind_pres
                    cv2.imshow("Frame", frame)
                    t0 = t1


if __name__ == "__main__":
    manager = Manager(scenes)

    load_thread = threading.Thread(target=manager.run)
    load_thread.start()
    manager.play()
    cv2.destroyAllWindows()


# i = 0
# paused = False
# direction = 1
# prev, prev_stop_frame, t0 = 0, 0, 0
# while True:
#     frame = frames[i]
#     cv2.imshow("Frame", frame)

#     delta = time.time() - t0
#     wait_time = max(1, int(1000 * (1 / FPS - delta))) * int(not paused)
#     key = cv2.waitKey(wait_time) & 0xFF

#     if key == ord(" "):
#         paused = not paused
#         prev_stop_frame = i
#     elif key == 81:
#         i = max(0, i - FPS // 2)
#     elif key == 85:
#         i = min(len(frames - 1), i + FPS // 2)
#     elif key == ord("q"):
#         break
#     elif key == ord("h"):
#         direction = -1
#     elif key == ord("l"):
#         direction = 1
#     elif (
#         not paused
#         and abs(i - prev_stop_frame) > FPS_SOURCE / 2
#         and np.all(frame == prev)
#         and False
#     ):
#         paused = True
#         prev_stop_frame = i

#     prev = frame
#     t0 = time.time()
#     i = max(0, min(len(frames) - 1, i + direction))
