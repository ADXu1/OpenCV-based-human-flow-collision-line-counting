# ----------------------------------------------------------------------------------------------------------------------
# 检测视频
# ----------------------------------------------------------------------------------------------------------------------

from utils.datasets import *
from utils.utils import *
import argparse
import imutils
import cv2

# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='path to weights file')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
opt = parser.parse_args()
print(opt)


# ----------------------------------------------------------------------------------------------------------------------


class Yolo():
    def __init__(self):
        self.writer = None
        self.prepare()

    def prepare(self):
        global model, device, classes, colors, names
        device = torch_utils.select_device(device='cpu')

        google_utils.attempt_download(opt.weights)
        model = torch.load(opt.weights, map_location=device)['model'].float()

        model.to(device).eval()

        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    def detect(self, frame):
        im0 = frame
        img = letterbox(frame, new_shape=416)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        boxes = []
        confidences = []
        classIDs = []

        for i, det in enumerate(pred):

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, score, cls in det:
                    label = '%s ' % (names[int(cls)])

                    if names[int(cls)] in ['person']:
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    # ----------------------------------------------------------------------------------------------------------------------

                    boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])])
                    confidences.append(float(score))
                    classIDs.append(int(cls))

            return im0


# ----------------------------------------------------------------------------------------------------------------------

yolo = Yolo()
writer = None
cap = cv2.VideoCapture('./inference/test_1.mp4')
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=720)
    frame = yolo.detect(frame)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("./inference/out_video.mp4",
                                 fourcc,
                                 20,
                                 (frame.shape[1], frame.shape[0]),
                                 True)

    writer.write(frame)

    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------------------------------------------------------------
