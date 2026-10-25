# from ultralytics import YOLO
# from common import main_loop, DEVICE
#
# model = YOLO("yolo26n.pt")
#
# def detect(frame):
#     r = model(frame, verbose=False, device=DEVICE)[0]
#     return [(*map(int, b.xyxy[0]), model.names[int(b.cls[0])]) for b in r.boxes]
#
# main_loop("YOLO26", detect)


from ultralytics import YOLO
from common import main_loop, DEVICE

# YOLO26 nano modelini yuklash
model = YOLO("yolo26n.pt")


def detect(frame):
    results = model.predict(
        source=frame,
        device=DEVICE,
        imgsz=640,
        conf=0.25,
        verbose=False
    )[0]

    boxes = []

    if results.boxes is not None:
        for b in results.boxes:
            x1, y1, x2, y2 = (
                b.xyxy[0].detach().cpu().numpy().astype(int)
            )

            class_id = int(b.cls[0].item())
            label = model.names[class_id]

            boxes.append((
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                str(label)
            ))

    return boxes


if __name__ == "__main__":
    main_loop("YOLO26", detect)
