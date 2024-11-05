from queue import Queue
from threading import Thread

from config import ConfigEuRoC
from image import ImageProcessor
from msckf_uwb import MSCKF
import scipy.io as sio



class VIRO(object):
    def __init__(self, config, img_queue, imu_queue, uwb_queue, gt_queue, viewer=None):

        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.gt_queue = gt_queue
        self.uwb_queue = uwb_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.gt_thread = Thread(target=self.process_gt)
        self.uwb_thread =Thread(target=self.process_uwb)
        self.vio_thread = Thread(target=self.process_feature)

        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()
        self.gt_thread.start()
        self.uwb_thread.start()

        # data save
        self.vio_data = []
        self.gt_data = []


    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            # print('img_msg', img_msg.timestamp)

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.stareo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)


    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                # save data in .mat
                save_data = [{'timestamp': data.timestamp, 'pose': data.pose, 'velocity': data.velocity,
                              'cam0_pose': data.cam0_pose, 'anchor_pos':data.anchor_pos} for data in self.vio_data]
                sio.savemat('results/MH_04_difficult_uwb.mat', {'data': save_data})

                return
            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)

            if result is not None and self.viewer is not None:
                self.viewer.update_pose(result.cam0_pose)
                self.viewer.update_anchor(result.anchor_pos)
                self.vio_data.append(result)

    def process_gt(self):
        while True:
            gt_msg = self.gt_queue.get()

            if gt_msg is None:
                # save_data = [{'timestamp': data.timestamp, 'pose': data.p,
                #               'velocity': data.v, 'orientation': data.q} for data in self.gt_data]
                # sio.savemat('results/MH_01_easy_gt.mat', {'data': save_data})
                return
            else:
                self.gt_data.append(gt_msg)

    def process_uwb(self):
        while True:
            uwb_msg = self.uwb_queue.get()
            if uwb_msg is None:
                return
            self.msckf.uwb_callback(uwb_msg)
            print('uwb_msg', uwb_msg.timestamp)



if __name__ == '__main__':
    import time
    import argparse

    from dataset import EuRoCDataset, DataPublisher
    from viewer_uwb import Viewer
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/easy/stereo_msckf/data/MH_04_difficult', help='Path of EuRoC MAV dataset.')
    parser.add_argument('--view', action='store_true', help='Show trajectory.')
    args = parser.parse_args()
    args.view = True

    if args.view:
        viewer = Viewer()
    else:
        viewer = None

    dataset = EuRoCDataset(args.path)
    # dataset.set_starttime(offset=40.)   # start from static state MH01
    # dataset.set_starttime(offset=10.)  # start from static state MH02
    # dataset.set_starttime(offset=1.)  # start from static state MH03
    dataset.set_starttime(offset=15.)  # start from static state MH04

    img_queue = Queue()
    imu_queue = Queue()
    gt_queue = Queue()
    uwb_queue = Queue()


    config = ConfigEuRoC()
    msckf_vio = VIRO(config, img_queue, imu_queue, uwb_queue, gt_queue, viewer=viewer)


    duration = float('inf')
    ratio = 0.8  # make it smaller if image processing and MSCKF computation is slow
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, ratio)
    uwb_publisher = DataPublisher(
        dataset.uwb, uwb_queue, duration)
    gt_publisher = DataPublisher(
        dataset.groundtruth, gt_queue, duration)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    uwb_publisher.start(now)
    gt_publisher.start(now)
