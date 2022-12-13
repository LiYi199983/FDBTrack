import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from FDBTrack import matching
from FDBTrack.basetrack import BaseTrack, TrackState
from FDBTrack.kalman_filter import KalmanFilter

from FDBTrack.fast_reid_interfece import FastReIDInterface    #外观模块要复刻一个






class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):
        #feat是目标外观特征
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.q = 0.9
        self.score = score
        self.smooth_score = 0
        self.tracklet_len = 0
        self.Nu = 0.1
        self.high_score = 0.6

        self.smooth_feat = None   #轨迹外观特征
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat, self.score)
        self.features = deque([], maxlen=feat_history)

        # self.alpha = 0.9

    def update_score(self, curr_score):
        if self.smooth_score == 0:
            self.smooth_score = curr_score
        elif curr_score <= self.high_score:
            self.smooth_score = self.smooth_score
        else:
            self.smooth_score = self.q * self.smooth_score + (1-self.q) * curr_score
        return self.smooth_score

    def update_features(self, feat, curr_score):   #外观模型更新操作
        cur_alpha = self.update_alpha(curr_score)
        feat /= np.linalg.norm(feat)  #先归一话
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = (1 - cur_alpha) * self.smooth_feat + cur_alpha * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  #用于求范数，归一化操作

    def update_alpha(self, curr_score):
        d_vul = curr_score - self.smooth_score
        if d_vul >= 0:
            cur_alpha = (1-self.Nu) * d_vul + self.Nu
        else:
            cur_alpha = self.Nu * d_vul + self.Nu
        return cur_alpha


    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

        self.smooth_score = self.update_score(new_track.score)
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)   #feat, ss,cs,Nu
        self.state = TrackState.Tracked
        self.is_activated = True
        new_track.smooth_score = self.update_score(new_track.score)
        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class FDBTrack(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh   #拒绝IOU小于某个阈值的匹配
        self.appearance_thresh = args.appearance_thresh  #外观匹配阈值
        self.weight_lambda = args.weight_lambda

        if args.with_reid:
            self.encoder = FastReIDInterface(args.model_type, args.fast_reid_weights, args.device, args.batch_size)    #需要更改这个类

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                # output_results = output_results.cpu().numpy()    #记得开这个，如果有问题的话
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)  #输入图片，和全部的高分检测，得出目标特征

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections)   #他除以了2
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = self.weight_lambda * ious_dists + (1-self.weight_lambda) * emb_dists

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:  #对匹配上的轨迹进行寿命管理
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
        else:
            dets_second = []
            scores_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]  #如果跟踪状态的轨迹中有没有匹配上的轨迹就把他放到池里
        dists = matching.iou_distance(r_tracked_stracks, detections_second)  #未匹配上的轨迹和低分检测匹配
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:   #轨迹状态管理
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)  #如果不是在跟踪状态，那就是重新激活这个轨迹
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:  #如果没匹配上的轨迹不是lost状态，就把他变成lost状态，并放到lost轨迹中
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]  #拿出第一次匹配没有匹配上检测
        ious_dists = matching.iou_distance(unconfirmed, detections)   #未确认轨迹和第一次没有匹配上的轨迹进行匹配
        ious_dists_mask = (ious_dists > self.proximity_thresh)   #设置IOU距离阈值
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections)    #加入外观匹配阶段
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = self.weight_lambda * ious_dists + (1-self.weight_lambda) * emb_dists
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)   #匈牙利算法
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)   #匹配上的未确定轨迹直接更新激活
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:    #没有匹配上的直接删除
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:    #高分检测中，还未匹配成功的直接生成新轨迹
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:  #丢失的轨迹状态更新
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
