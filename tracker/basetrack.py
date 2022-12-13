import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0  #新轨迹
    Tracked = 1 #跟踪状态的
    Lost = 2  #失去跟踪的
    Removed = 3 #删除的


class BaseTrack(object):
    _count = 0  #计数器

    track_id = 0 #轨迹编号由0开始
    is_activated = False #激活开始关闭
    state = TrackState.New #开始状态都是新轨迹

    history = OrderedDict() #历史状态是一个有序的字典
    features = []  #特征池一开始是个空的列表池
    curr_feature = None  #当前提取的特征，设为None，就是不利用
    score = 0  #分数初始化为0
    start_frame = 0  #开始帧为0
    frame_id = 0 #帧号开始为0
    time_since_update = 0  #距离上次更新的时间设置为0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args): #激活轨迹是一个列表
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):  #标记失去跟踪的
        self.state = TrackState.Lost

    def mark_removed(self):  #标记移除的
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0