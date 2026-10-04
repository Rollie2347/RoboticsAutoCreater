"""
list_cameras.py

Run this first. It opens each camera index in turn (0 through MAX_INDEX-1)
and shows a live preview labeled with its index, so you can figure out
which physical webcam is which before setting CAMERA_IDS in config.py.

Controls:
    n / SPACE  -> move to the next index
    q / ESC    -> quit
"""
import sys

import cv2

MAX_INDEX = 2  # scans indices 0..MAX_INDEX-1; raise this if you have more cameras


def probe_index(index: int) -> bool:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"[index {index}] could not open")
        cap.release()
        return False

    print(f"[index {index}] opened -- showing preview, press n/SPACE for next, q to quit")
    found_next = False
    quit_all = False
    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"[index {index}] opened but failed to read a frame")
            break

        label = f"Camera index {index}  (n/SPACE = next, q = quit)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("list_cameras", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("n"), ord(" ")):
            found_next = True
            break
        if key in (ord("q"), 27):
            quit_all = True
            break

    cap.release()
    if quit_all:
        cv2.destroyAllWindows()
        sys.exit(0)
    return found_next


def main():
    print("Scanning camera indices 0.." + str(MAX_INDEX - 1))
    for i in range(MAX_INDEX):
        probe_index(i)
    cv2.destroyAllWindows()
    print("Done. Set CAMERA_IDS in config.py to the three indices you want to use, in order.")


if __name__ == "__main__":
    main()
