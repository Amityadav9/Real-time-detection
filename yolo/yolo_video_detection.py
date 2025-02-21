from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model("Studio_Test_01.mp4", save=True, show=True)
