# ----------------------------------------------------------------------------------------------------------------------
# 导包
# ----------------------------------------------------------------------------------------------------------------------

from tools import generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import preprocessing
from deep_sort import nn_matching
from collections import deque
import numpy as np
import detect
import cv2


# ----------------------------------------------------------------------------------------------------------------------

writer = None
cap = cv2.VideoCapture('./inference/test_1.mp4')

# ----------------------------------------------------------------------------------------------------------------------
# pts --- 放置点的容器
# np.random.seed(100) --- 随机数
# COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8") --- 设置随机颜色
# ----------------------------------------------------------------------------------------------------------------------

pts = [deque(maxlen=30) for _ in range(9999)]
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# ----------------------------------------------------------------------------------------------------------------------
# deepsort相关参数
# ----------------------------------------------------------------------------------------------------------------------

max_cosine_distance = 0.3
nms_max_overlap = 1.0
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
encoder = gdet.create_box_encoder('./weights/mars-small128.pb', batch_size=1)
tracker = Tracker(metric)

# ----------------------------------------------------------------------------------------------------------------------
# line = [(110, 100), (460, 200)] --- 线的设置，两端的点
# counter_sum counter_up counter_down --- 三个计数器
# ----------------------------------------------------------------------------------------------------------------------

line = [(110, 100), (460, 200)]
counter_sum = 0
counter_up = 0
counter_down = 0

# ----------------------------------------------------------------------------------------------------------------------
# 行人计数器
# ----------------------------------------------------------------------------------------------------------------------

counter_person = 0


# ----------------------------------------------------------------------------------------------------------------------
# 下面两个函数用于得到方框和线的关系 --- 类似于碰撞检测
# ----------------------------------------------------------------------------------------------------------------------

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# ----------------------------------------------------------------------------------------------------------------------


while True:
    ret, frame = cap.read()

    if not ret:
        break

    i = int(0)

    # ----------------------------------------------------------------------------------------------------------------------
    # 调用detect.py里面的recognition()函数，
    # 返回数据为：[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(score), int(cls), label]
    # 左上角的横坐标，左上角的纵坐标，右下角的横坐标，右下角的纵坐标，置信度，类别序号，类别
    # ----------------------------------------------------------------------------------------------------------------------

    boxs_ = detect.recognition(frame)
    boxes = []

    # ----------------------------------------------------------------------------------------------------------------------
    # 对返回的数据[[], [], []...]进行遍历，然后加入到boxes里面 --> deepsort
    # ----------------------------------------------------------------------------------------------------------------------

    for box in boxs_:
        if box[6] == 'person ':
            boxes.append([int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])])

    # ----------------------------------------------------------------------------------------------------------------------

    features = encoder(frame, boxes)
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)
    indexIDs = []

    # ----------------------------------------------------------------------------------------------------------------------
    # 对deepsort处理的数据进行遍历
    # ----------------------------------------------------------------------------------------------------------------------

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        indexIDs.append(int(track.track_id))
        bbox = track.to_tlbr()
        color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
        # cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 5e-3 * 150, (color), 2)

        i += 1
        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        pts[track.track_id].append(center)
        thickness = 5
        cv2.circle(frame, center, 1, color, thickness)

        # ----------------------------------------------------------------------------------------------------------------------
        # if len(pts[track.track_id]) > 3: --- 判断跟踪物体轨迹中点的数量
        # ----------------------------------------------------------------------------------------------------------------------

        if len(pts[track.track_id]) > 3:
            cv2.line(frame, (pts[track.track_id][-2]), (pts[track.track_id][-1]), color, thickness)

            point_previous = pts[track.track_id][-2]
            point_present = pts[track.track_id][-1]

            # ----------------------------------------------------------------------------------------------------------------------
            # 将一个物体上一帧的中心点和当前帧的中心点进行连线，之后和设置的点进行碰撞检测，如果返回True，则有车辆通过
            # ----------------------------------------------------------------------------------------------------------------------

            if intersect(point_previous, point_present, line[0], line[1]):
                counter_sum += 1

                # ----------------------------------------------------------------------------------------------------------------------
                # 判断物体上一帧的中心点的y值和当前帧的中心点的y值大小，判断车流的方向
                # ----------------------------------------------------------------------------------------------------------------------

                if point_present[1] > point_previous[1]:
                    counter_down += 1
                else:
                    counter_up += 1

    # ----------------------------------------------------------------------------------------------------------------------
    # cv2.line(frame, line[0], line[1], (0, 255, 0), 3) --- 绘制基准线，根据自己的实际的参数，选择
    # cv2.putText --- 左上角计数数据的显示
    # str(counter) --- counter是整数类型的数据，必须要转为字符型的数据才可以显示
    # cv2.putText --- 参数 --- 当前帧，显示内容，显示位置，字体类型，字体大小，字体颜色，字体胖瘦（哈哈哈）
    # ----------------------------------------------------------------------------------------------------------------------

    cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
    cv2.putText(frame, str(counter_sum), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)
    cv2.putText(frame, str(counter_up), (230, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 0), 3)
    cv2.putText(frame, str(counter_down), (430, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 3)

    # ----------------------------------------------------------------------------------------------------------------------
    # 对一帧中行人的数量进行显示，然后将计数器初始化，再进行下一帧的计数，
    # ----------------------------------------------------------------------------------------------------------------------

    # cv2.putText(frame, 'person: ' + str(counter_person), (630, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 3)

    # ----------------------------------------------------------------------------------------------------------------------
    # 将处理的结果写成视频文件
    # ----------------------------------------------------------------------------------------------------------------------

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("./inference/out.mp4",
                                 fourcc,
                                 20,
                                 (frame.shape[1], frame.shape[0]),
                                 True)

    writer.write(frame)

    # ----------------------------------------------------------------------------------------------------------------------
    # 使用opencv显示处理后的结果
    # ----------------------------------------------------------------------------------------------------------------------

    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------------------------------------------------------------