#!/usr/bin/env python3
"""
Driving object detection quick start (YOLOv8n, laptop).

Usage:
    python driving_detect.py setup                    # install deps + download weights
    python driving_detect.py run                      # webcam
    python driving_detect.py run --source dashcam.mp4 # video file
    python driving_detect.py run --imgsz 640          # higher accuracy, slower
    python driving_detect.py export                   # TFLite export for the Pixel
    python driving_detect.py export --int8            # INT8 (faster on Tensor chip)

Press 'q' in the video window to quit.
"""
import argparse
import subprocess
import sys

# COCO class IDs relevant to driving
DRIVING_CLASSES = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    11,  # stop sign
]


def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


def cmd_setup(args):
    print("Installing dependencies...")
    pip_install("ultralytics", "opencv-python")
    print("Downloading YOLOv8n weights...")
    from ultralytics import YOLO
    YOLO("yolov8n.pt")  # auto-downloads to the working directory
    print("\nSetup done. Try:  python driving_detect.py run")


def cmd_run(args):
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    source = int(args.source) if args.source.isdigit() else args.source
    classes = None if args.all_classes else DRIVING_CLASSES

    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        classes=classes,
        stream=True,
        verbose=False,
    )

    import cv2
    for r in results:
        cv2.imshow("Driving detection (q to quit)", r.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def cmd_export(args):
    # TFLite needs extra packages; Ultralytics installs most of them on demand.
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    kwargs = dict(format="tflite", imgsz=args.imgsz)
    if args.int8:
        kwargs.update(int8=True, data="coco8.yaml")
    else:
        kwargs.update(half=True)
    path = model.export(**kwargs)
    print(f"\nExported to: {path}")
    print("Copy the .tflite file to your Pixel and load it in an Android app.")


def main():
    p = argparse.ArgumentParser(description="YOLOv8n driving detection quick start")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="install dependencies and download weights").set_defaults(func=cmd_setup)

    r = sub.add_parser("run", help="run detection on webcam or video")
    r.add_argument("--source", default="0", help="webcam index (0) or path to video/image")
    r.add_argument("--imgsz", type=int, default=320, help="input size (320 fast, 640 accurate)")
    r.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    r.add_argument("--all-classes", action="store_true", help="detect all 80 COCO classes")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("export", help="export model to TFLite for Android")
    e.add_argument("--imgsz", type=int, default=320)
    e.add_argument("--int8", action="store_true", help="INT8 quantization")
    e.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
