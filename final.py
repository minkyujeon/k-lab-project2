import cv2
import gi

from time import clock

import socket
import select
import numpy as np
import sys
import re
import hashlib
import base64
import threading
import struct
import subprocess
from queue import Queue

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject


# python run_video.py --model=mobilenet_v2_large --video=
import argparse
import logging
import time

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# object detection
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv



IP = ''
PORT_PI = 8485          # Video Input Port
PORT_ANDROID = 3000     # Video Output Port
SIZE = 1024

ADDR_PI = (IP, PORT_PI)             # Video Input address
ADDR_ANDROID = (IP, PORT_ANDROID)   # Video Output address

in_queue = Queue()      # Data Queue Shared with Mobile
openposeBeforeQ = Queue()
detectBeforeQ = Queue()

openposeAfterQ = Queue()
detectAfterQ = Queue()

##################### Gstreamer #####################
### Factory
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
#        self.cap = cv2.VideoCapture("./BigBuckBunny1.mp4")
        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=512,height=512,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)

    def on_need_data(self, src, lenght):

        frame = in_queue.get()  # Frame from queue

        data = frame.tostring()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = timestamp
        self.number_frames += 1
        retval = src.emit('push-buffer', buf)
        print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                       self.duration,
                                                                                       self.duration / Gst.SECOND))
        if retval != Gst.FlowReturn.OK:
            print(retval)

        in_queue.task_done()

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

### UDP Server
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)
        self.attach(None)
#####################################################



def recvall(sock, count):
    # Byte String
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

### Thread: input video
def handlePI(client_socket):
    # Size of StringData from clinet: (==(str(len(stringData))).encode().ljust(16))
    while True:
        # Bytes String to image frame
        length = recvall(client_socket, 16)
        stringData = recvall(client_socket, int(length))
        temp = np.fromstring(stringData, dtype = 'uint8')
        frame = cv2.imdecode(temp, cv2.IMREAD_COLOR)

        #######################
        #  Model will be here #
        #######################
        
        in_queue.put(frame)
        in_queue.join()

        openposeBeforeQ.put(frame)
        openposeBeforeQ.join()

        detectBeforeQ.put(frame)
        detectBeforeQ.join()

### Thread: output video
def handleAndroid(client_socket):
    GObject.threads_init()
    Gst.init(None)

    server = GstServer()

    loop = GObject.MainLoop()
    loop.run()

##################### Server Main #####################

### Socket
# PI - SERVER
server_socket_PI = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket_PI.settimeout(1)
server_socket_PI.bind(ADDR_PI)
server_socket_PI.listen()

# ANDROID - SERVER
server_socket_Android = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket_Android.settimeout(1)
server_socket_Android.bind(ADDR_ANDROID)
server_socket_Android.listen()

read_socket_list = [server_socket_PI, server_socket_Android]

print("Server is Running")
print("Port")
print("Raspberry Pi : 8485")
print("Android : 3000")

### Main server run
# while True:
#     # Listen socket
#     conn_read_socket_list, conn_write_socket_list, conn_except_socet_list = select.select(read_socket_list, [], [])

#     for conn_read_socket in conn_read_socket_list:

#         # Raspberry PI - Server socket
#         if conn_read_socket == server_socket_PI:
#             print('PI Connected')
#             client_socket, client_addr = server_socket_PI.accept()
#             PI_handler = threading.Thread(
#                 target = handlePI,
#                 args = (client_socket,)
#             ).start()

#         # Android - Server socket
#         elif conn_read_socket == server_socket_Android:
#             print('Android Connected')
#             client_socket2, client_addr2 = server_socket_Android.accept()
#             Android_handler = threading.Thread(
#                 target = handleAndroid,
#                 args = (client_socket2,)
#             ).start()

def piConnected():
    while True:
        client_socket, client_addr = server_socket_PI.accept()
        print('PI Connected')
        PI_handler = threading.Thread(
            target = handlePI,
            args = (client_socket,)
        ).start()

def androidConnected():
    while True:
        client_socket2, client_addr2 = server_socket_Android.accept()
        print('Android Connected')
        Android_handler = threading.Thread(
            target = handleAndroid,
            args = (client_socket2,)
        ).start()

##################### OpenPose Main #####################

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def openposeMain():
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    # parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='512x512', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    while True:
        image = openposeBeforeQ.get()

        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)

        # print('len(humans):',len(humans)) #frame하나당 사람의 좌표
        if not args.showBG:
            image = np.zeros(image.shape)
        (image,hands_centers) = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        # return value
        openposeAfterQ.put(hands_centers)

        num_hands = len(hands_centers)

        # if num_hands != 0:
        #     for i in range(num_hands):
        #         print('i:',i,'hands_centers_inin:',hands_centers[i])
        
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        # if cv2.waitKey(1) == 27:
        #     break

    # cv2.destroyAllWindows()
### openposeMain End

logger.debug('finished+')


##################### Detection Main #####################

config_file = '/home/Downloads/k-lab/ttfnet/configs/ttfnet/ttfnet_d53_2x.py'
checkpoint_file = '/home/Downloads/k-lab/ttfnet/pretrain/epoch_24.pth'

class PersonNObjBboxes(person_bboxes, object_bboxes):
    def __init__(self, **properties):
        super(PersonNObjBboxes, self).__init__(**properties)
        self.person_bboxes = person_bboxes
        self.object_bboxes = object_bboxes

def detectionMain():
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # video = mmcv.VideoReader('degrade.mov')
    # https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/io.py

    while True:
        frame = detectBeforeQ.get()
        result = inference_detector(model, frame) #bbox result

        (person_bboxes, object_bboxes) = show_result(frame, result, model.CLASSES, wait_time=2)
        print('person:',person_bboxes)
        print('object:',object_bboxes)

        returnVal = PersonNObjBboxes(person_bboxes, object_bboxes)
        detectAfterQ.put(returnVal)

if __name__ == '__main__':
    thread_pi = threading.Thread(target = piConnected, daemon = True)
    thread_android = threading.Thread(target = androidConnected, daemon = True)
    thread_openpose = threading.Thread(target = openposeMain, daemon = True)
    thread_detect = threading.Thread(target = detectionMain, daemon = True)

    thread_pi.start()
    thread_android.start()
    thread_openpose.start()
    thread_detect.start()