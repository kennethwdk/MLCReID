import os.path as osp
from glob import glob
import re

class DataSet(object):
    def __init__(self, data_dir, name='makret', info=True):
        self.name = name
        self.images_dir = osp.join(data_dir, name)
        self.train_path = 'bounding_box_train'
        self.train_camstyle_path = 'bounding_box_train_camstyle'
        self.query_path = 'query'
        self.gallery_path = 'bounding_box_test'

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.cam_dict = self.set_cam_dict()
        self.num_cam = self.cam_dict[name]

        self.load(info)

    def set_cam_dict(self):
        cam_dict = {}
        cam_dict['market'] = 6
        cam_dict['duke'] = 8
        cam_dict['msmt17'] = 15
        return cam_dict

    def preprocess(self, images_dir, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        idx2pid = []
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        cnt = 0
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam, cnt))
            idx2pid.append(pid)
            cnt = cnt + 1
        if relabel:
            return ret, int(len(all_pids)), idx2pid
        else:
            return ret, int(len(all_pids))

    def load(self, info=True):
        self.train, self.num_train_ids, self.idx2pid = self.preprocess(self.images_dir, self.train_path)
        self.query, self.num_query_ids = self.preprocess(self.images_dir, self.query_path, relabel=False)
        self.gallery, self.num_gallery_ids = self.preprocess(self.images_dir, self.gallery_path, relabel=False)

        if info:
            print(self.__class__.__name__, self.name, "loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | 'Unknown' | {:8d}"
                .format(len(self.train)))
            print("  query    | {:5d} | {:8d}"
                .format(self.num_query_ids, len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                .format(self.num_gallery_ids, len(self.gallery)))