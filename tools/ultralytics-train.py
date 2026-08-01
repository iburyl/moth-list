from ultralytics import YOLO

#model = YOLO("yolo11n-pose.pt")
model = YOLO("yolo26s-pose.pt")

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
    project="runs/moth_pose",
    name="initial",
)
