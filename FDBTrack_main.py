import cv2 as cv
import argparse
from yolox.my_yolo import YoloX
from yolox.utils.val_dataloader import LoadImages
from FDBTrack.FDB_track import FDBTrack
from tracker.timer import Timer
from yolox.utils.visualize import plot_tracking, increment_path
import os
import pathlib as Path
from loguru import logger
import time


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_file', default=,   #yolo_exp path
                        help='exp path')
    parser.add_argument('--weight_path',
                        default=,    #yolo weight path
                        help='weight path')
    parser.add_argument("--save", default=True, help='save results')
    parser.add_argument('--save_txt', default=True, help='save txt results')
    parser.add_argument('--show', default=True, help='show frame img')
    opt = parser.parse_args()
    return opt

def parse_args():
    parser = argparse.ArgumentParser()
    # tracking args
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float,
                        help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--model_type", dest="model_type", default="osnet_pcb_x1_0",
                        type=str, help="reid model  osnet_pcb_x1_0  |  osnet_x1_0  | ckpt | resnest50 | shufflenetv2_x2_0")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=,    #re-id model weight path
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.4,
                        help='threshold for rejecting low appearance reid matches')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--batch_size", default=250, help="re-id model batch_size")
    parser.add_argument('--weight_lambda', default=0.9, help='Fusion weight of IOU distance and embedded distance')
    args = parser.parse_args()
    return args

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                          w=round(w, 1), h=round(h, 1), s=round(score, 2))#舍去小数点后1位数字
                f.write(line)
    logger.info('save results to {}'.format(filename))


def main(i):
    opt = parse_opt()
    args = parse_args()
    data_path = 'dataset/train/MOT17-{}-FRCNN\img1'.format(i)
    save_dir = 'outdata/runs/FDBTrack/MOT17-{}-FRCNN'.format(i)
    eval_path = 'outdata/runs/FDBTrack/MOT17-{}-FRCNN'.format(i)
    yolo = YoloX(opt.exp_file, opt.weight_path)
    tracker = FDBTrack(args)
    print(data_path)
    dataset = LoadImages(data_path, img_size=yolo.test_size)

    timer = Timer()
    results = []
    frame_id = 0
    vid_path, vid_writer = [None] * 1, [None] * 1
    desktop_path = eval_path
    filename = desktop_path + '.txt'
    if opt.save:
        # Directories
        save_dir = increment_path(save_dir)  # increment run
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    for path, im, im0s, vid_cap, s in dataset:
        frame_id += 1
        height, width = im0s.shape[:2]
        outputs = yolo.detect_bounding_box(im, im0s)
        scale = min(800 / float(height, ), 1440 / float(width))  # image size [h, w]
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale
            online_targets = tracker.update(detections, im0s)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

        timer.toc()
        online_im = plot_tracking(
            im0s, online_tlwhs, online_ids, online_scores, frame_id=frame_id, fps=1. / timer.average_time
        )
        if opt.show:
            cv.imshow('img', online_im)
            cv.waitKey(30)

        if opt.save:
            image_name = path[path.rfind('\\') + 1:]
            save_path = os.path.join(save_dir, image_name)
            if dataset.mode == 'image':
                cv.imwrite(save_path, online_im)
            else:
                if vid_path[0] != save_path:  # new video
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv.VideoWriter):
                        vid_writer[0].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    vid_writer[0] = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0s)

    write_results(filename=filename, results=results)
    print("the {} epoch is over !".format(i))





if __name__ == '__main__':
    main('05')