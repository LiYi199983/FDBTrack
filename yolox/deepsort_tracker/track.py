# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.#轨迹在确认前的连续匹配次数，根据这个来删除误检轨迹
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.#已经确认的轨迹有连续多少帧没有被匹配上
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.#对于没确认的轨迹，就是none，确认的轨迹就是一个特征描述子池
    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.一共成功匹配上了多少次
    age : int
        Total number of frames since first occurance.由初始帧开始，存在了的年龄
    time_since_update : int
        Total number of frames since last measurement update.自从上一次的测量更新以来，一个过去了多少帧
    state : TrackState
        The current track state.#当前轨迹处于什么状态
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.一个缓存特征向量的列表，就是一个池子
    """

    def __init__(self, mean, covariance, track_id, class_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.class_id = class_id #目标的类别索引
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def increment_age(self):#增加轨迹的寿命
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment_age()#预测一下，然后轨迹的寿命+1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed#暂时轨迹转正

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).删除轨迹，进行轨迹寿命的管理
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted