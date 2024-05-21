from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
if __name__ == '__main__':
    results = model.train(data="E:\homework\MVClab\week4\k-street-food\data.yaml", epochs=100, imgsz=640)

