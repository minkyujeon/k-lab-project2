from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = '/home/Downloads/k-lab/ttfnet/configs/ttfnet/ttfnet_d53_2x.py'
checkpoint_file = '/home/Downloads/k-lab/ttfnet/pretrain/epoch_24.pth'

def main(config_file, checkpoint_file):
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    video = mmcv.VideoReader('degrade.mov')
    # https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/io.py

    for frame in video:
        result = inference_detector(model, frame) #bbox result
        # print('result[0]:',result[0])
        # print('len(result):',len(result))
        # print('model.CLASSES:',model.CLASSES)
        show_result(frame, result, model.CLASSES, wait_time=2)
        
if __name__ == '__main__':
    main(config_file, checkpoint_file)