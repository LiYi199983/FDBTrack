# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    Parameters
    ----------
    tlwh : array_like #框左上角点和宽高
        Bounding box in format `(x, y, w, h)`.
    confidence : float#置信度
        Detector confidence score.
    feature : array_like#一个数组，外观描述子
        A feature vector that describes the object contained in this image.
    Attributes#数据的属性
    ----------
    tlwh : ndarray#多维数组
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray#多维数组
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):   #将tlwh格式转化为左上右下格式
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):  #将tlwh格式转成deepsort论文规定的格式
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret