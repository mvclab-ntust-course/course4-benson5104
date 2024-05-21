from ultralytics import YOLO

# Load a model
model = YOLO("E:\homework\資訊安全導論\pythonProject2\MVCweek4\\runs\detect\\train16\weights\\best.pt")

# Customize validation settings
if __name__ == '__main__':
    validation_results = model.val(data="E:\homework\MVClab\week4\k-street-food\data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0",split = "test")