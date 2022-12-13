# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import  #这是为了引入绝对库，而不是经过改动的本目录下的同名库
import numpy as np
from yolox.deepsort_tracker import linear_assignment  #导入线性指派文件


def iou(bbox, candidates):

    #输入一个1*4轨迹预测框，n*4检测候选框，输出一个n*1的IOU
    """Computer intersection over union.
    Parameters
    ----------
    bbox : ndarray   左上宽高格式的包围框 是一个1行4列的多维数组
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray 没行一个的矩阵，每行代表一个候选框的左上宽高格式  n*4的矩阵
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.
    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate. #更高的分数意味着更大面积的的bbox与候选框重合
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]#转为左上右下格式
    candidates_tl, candidates_br = candidates[:, :2], candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]#n*2的数组，为交区域左上点坐标
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]#n*2数组，为交区域右上点坐标
    wh = np.maximum(0., br - tl) #n*2数组，交区域宽高为正值

    area_intersection = wh.prod(axis=1)#交区域面积为n*1
    area_bbox = bbox[2:].prod()#框面积。为1*1
    area_candidates = candidates[:, 2:].prod(axis=1)#候选框面积。为n*1
    return area_intersection / (area_bbox + area_candidates - area_intersection) #n*1，反映检测框与候选框间的IOU


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.
    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.#一个轨迹列表，由Track输出
    detections : List[deep_sort.detection.Detection]
        A list of detections.#一个检测列表由Detection管理，其中可包含外观描述子
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.#参与匹配的轨迹的索引列表，默认情况是所有轨迹都参与匹配
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.#参与匹配的检测的索引列表，默认是所有检测都参与匹配
    Returns
    -------
    ndarray
        Returns a cost matrix of shape#输出一个n*m的矩阵，每个元素为对应位置的IOU距离
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))  #输出一个[1,...n-1]的数组
    if detection_indices is None:
        detection_indices = np.arange(len(detections))  #输出一个[1,...m-1]的数组

    cost_matrix = np.zeros((len(track_indices), len(detection_indices))) #构造一个n*m的零矩阵
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST  #如果轨迹不是连续的，则按照INFTY计算该轨迹对应的行的匹配成本，就是调到无穷大的成本，保证匹配不上
            continue

        bbox = tracks[track_idx].to_tlwh()#如果是正常连续的先把轨迹预测框装成左上高宽的格式，这个函数是轨迹类自带的函数
        candidates = np.asarray(
            [detections[i].tlwh for i in detection_indices])#把检测框的坐标转化为tlwh格式的列表
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix