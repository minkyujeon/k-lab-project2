from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = '/home/Downloads/k-lab/ttfnet/configs/ttfnet/ttfnet_d53_2x.py'
checkpoint_file = '/home/Downloads/k-lab/ttfnet/pretrain/epoch_24.pth'

def main(config_file, checkpoint_file):
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    # img = 'image11945.jpg'  # or img = mmcv.imread(img), which will only load it once
    # result = inference_detector(model, img)
    # show_result(img, result, model.CLASSES)

    # # test a list of images and write the results to image files
    # imgs = ['test1.jpg', 'test2.jpg']
    # for i, result in enumerate(inference_detector(model, imgs)):
    #     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

    # test a video and show the results
    video = mmcv.VideoReader('IMG_8209.MOV') #('video.mp4')
    # https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/io.py
    # print('len(video):',len(video)) #129 - 5초 / 74 - 2초

    for frame in video:
        result = inference_detector(model, frame)
        # print('result[0]:',result[0])
        # print('len(result):',len(result))
        show_result(frame, result, model.CLASSES, wait_time=2)
        
if __name__ == '__main__':
    main(config_file, checkpoint_file)