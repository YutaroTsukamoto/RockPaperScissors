from roboflow import Roboflow
from ultralytics import YOLO

# 1. データセットのダウンロード
# ↓ あなたのAPIキーに書き換えてください
rf = Roboflow(api_key="JmsXeDZtWzGp2nOpfYUl")

project = rf.workspace("kkk-0qgmd").project("janken-cqbkx-puevy")
version = project.version(1)
dataset = version.download("yolov8")

# 2. 学習の実行
model = YOLO('yolov8n.pt')

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=30,
    imgsz=640
)