import argparse
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Train a YOLO model.")
parser.add_argument(
    "model",
    help="Model file to start from, e.g. yolo26s-pose.pt / yolo26s.pt",
)
parser.add_argument(
    "--config",
    default="yolo26s-pose.yaml",
    help="Model architecture YAML to build, e.g. yolo26s-pose.yaml / yolo26s.yaml",
)
parser.add_argument(
    "--project",
    default="pose_classification",
    help=(
        "Output parent directory (results go in <project>/<name>)."
    ),
)
parser.add_argument(
    "--name",
    default="init",
    help="Run subfolder inside --project (default: initial).",
)
args = parser.parse_args()

model = YOLO(args.config)
model.load(args.model)

# Absolute path avoids Ultralytics prepending its default runs/<task> dir.
project = str(Path(args.project).resolve())

model.train(
    data="data.yaml",
    epochs=150,

    imgsz=768,
    batch=8,
    device=0,
    workers=8,

    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=5,

    degrees=180,
    translate=0.05,
    scale=0.20,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.0,
    mixup=0.0,
    project=project,
    name=args.name,
)
