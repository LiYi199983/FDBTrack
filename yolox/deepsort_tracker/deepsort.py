import numpy as np
import torch
import cv2
import os

from .reid_model import Extractor
from yolox.reidmodel.osent_PCB import MYExtractor
from yolox.reidmodel.osnet import OSNETExtractor
from yolox.reidmodel.resnest import RESNESTExtractor
from yolox.reidmodel.shufflenetv2 import SHUFFLENETExtractor
from yolox.deepsort_tracker import kalman_filter, linear_assignment, iou_matching
from yolox.data.dataloading import get_yolox_datadir  #根据环境变量来指定yolox的权重
from .detection import Detection
from .track import Track


def _cosine_distance(a, b, data_is_normalized=False):#求外观的余弦距离
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)#按行向量求范式，并保持二维的性质
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)#在列方向求出最小的余弦距离


class Tracker:  #跟踪器
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric  #度量的方法
        self.max_iou_distance = max_iou_distance  #最大IOU距离
        self.max_age = max_age  #年龄
        self.n_init = n_init  #考察期限

        self.kf = kalman_filter.KalmanFilter()  #卡尔曼滤波器
        self.tracks = [] #装轨迹的列表
        self._next_id = 1 #下一次的新目标ID加多少

    def predict(self):  #轨迹具备预测功能，能通过卡尔曼滤波预测
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):   #增加年龄
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections, classes):  #轨迹的更新过程
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections) #运行级联匹配，得出3个集合

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])#更新匹配上的轨迹的状态
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()  #对没匹配上的轨迹考虑是否删除
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], classes[detection_idx].item()) #对于没匹配上的检测初始化
        self.tracks = [t for t in self.tracks if not t.is_deleted()] #删除掉被删除的轨迹，在轨迹集合中

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]#激活状态的目标
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):  #级联匹配

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, class_id):  #轨迹的初始化
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, class_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1


class NearestNeighborDistanceMetric(object):#最近邻度量
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "cosine": #度量距离的类型
            self._metric = _nn_cosine_distance  #在一个列表里选给最小的值出来
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold  #匹配阈值
        self.budget = budget  #池的深度，就是最近的100次
        self.samples = {}  #空字典，装样品的

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix  #输出的是余弦距离的度量矩阵，deepsort的第一个级联匹配是只用外观特征来进行的


class DeepSort(object):
    def __init__(self, model_path,
                 max_dist=0.4,   #0.1
                 min_confidence=0.5,  #0.5
                 nms_max_overlap=1.0,  #1.0
                 max_iou_distance=0.7, #0.7
                 max_age=30,  #30
                 n_init=3,  #3
                 nn_budget=100, #100
                 use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        # self.extractor = Extractor(model_path, use_cuda=use_cuda)#只要再给定一个特征提取器模型权重的位置就可以了
        self.extractor = MYExtractor(model_path, use_cuda=use_cuda)
        # self.extractor = OSNETExtractor(model_path, use_cuda=use_cuda)
        # self.extractor = SHUFFLENETExtractor(model_path, use_cuda=use_cuda)
        # self.extractor = RESNESTExtractor(model_path, use_cuda=use_cuda)


        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, output_results, img_info, img_size, img):
        # img_file_name = os.path.join(get_yolox_datadir(), 'mot', 'train', img_file_name)
        # ori_img = cv2.imread(img_file_name)  #路径和标志
        self.height, self.width = img.shape[:2]   #ori_img.shape[:2]
        # post process detections
        output_results = output_results.cpu().numpy()
        confidences = output_results[:, 4] * output_results[:, 5] #置信度是yolox后两位输出的积
        
        bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]#输入图像的高宽信息
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))#通过最小的比例，确定图像的尺度是多少
        bboxes /= scale #缩放包围框的信息，根据尺度
        bbox_xyxy = bboxes
        bbox_tlwh = self._xyxy_to_tlwh_array(bbox_xyxy) #转为tlwh格式
        remain_inds = confidences > self.min_confidence #只保留大于最小可信度阈值的目标
        bbox_tlwh = bbox_tlwh[remain_inds]
        confidences = confidences[remain_inds]

        # generate detections
        features = self._get_features(bbox_tlwh,img) #获取检测的外观特征
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence] #生成检测，包括位置、分数和外观特征
        classes = np.zeros((len(detections), ))

        # run on non-maximum supression #运行非极大值抑制，防止一个目标对应多个检测框，实际不需要这个东西
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict() #轨迹预测，轨迹更新
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy_noclip(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int)) #输出为包围框位置，轨迹号和类别号
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)#向列方向堆积
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh
    
    @staticmethod
    def _xyxy_to_tlwh_array(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy_noclip(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):  #组合全部的检测框位置，提取外观特征，并保留
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    # def _get_features(self, detect_boxes, img):
    #     # 输入detect_boxes形式为xyxy
    #     im_crops = []
    #     for box in detect_boxes:
    #         x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3]),
    #         im = img[y1:y2, x1:x2]
    #         im_crops.append(im)
    #     if im_crops:
    #         # 将划分的图片送入特征网络
    #         features = self.extractor(im_crops)
    #     else:
    #         features = np.array([])
    #     return features
