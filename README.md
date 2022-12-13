# FDBTRACK

## Code introduction and use

### Code
Please download YOLOX weight and reid network weight before use.
The ownership weight is in  (https://pan.baidu.com/s/1H3LH7k2XLLcWiIRxks0DCg?pwd=ly83)

The yolox folder contains the relevant code under YOLOX target tracking, 
the exp file under yolox is the model related configuration, the weights
under yolox need to be placed in the relevant weight file, and the dataset
folder under yolox needs to be placed in the picture sequence or video to
be detected and tracked

In the tracker folder, there are contents related to the bytetrack algorithm.
The FDBTrack folder contains the related content of the bytetrack algorithm

# use
This code can be run once through the main.py file. The parameter configuration
of this code is defined in main.py and can be modified by yourself.

## result
### Results on MOT challenge test set
| Dataset    |  MOTA | IDF1 | HOTA | FP | FN | IDs | matching speed (FPS) |
|------------|-------|------|------|------|------|------|------|
|MOT17       | 79.0 | 77.5 | 62.8 | 19383 | 96840 | 2394 | 46.3 |
|MOT20       | 75.9 | 75.9 | 62.0 | 21423 | 101874 | 1452 | 11.9 |


### Demo link
https://www.bilibili.com/video/BV1j24y1Q7UE/






