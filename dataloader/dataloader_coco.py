import os
from torch.utils.data import Dataset
import numpy as np
import clip
from PIL import Image
from pycocotools.coco import COCO
import json
import math

class MSCOCO_Dataset(Dataset):
    def __init__(self, args, image_root, annFile, preprocess, ids=None, subset='train', logger=None):
        logger.info("========== Initial the %s set ==========", subset)
        self.args = args
        self.image_root = image_root
        self.preprocess = preprocess
        self.subset = subset
        self.num_anns = 5

        self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        self.captions = [self.coco.loadAnns(annotation_id.item())[0]['caption'] for annotation_id in self.ids]
        logger.info('%d captions have been loaded.', len(self.captions))
        self.images_id = [self.coco.loadAnns(annotation_id.item())[0]['image_id'] for annotation_id in self.ids]
        self.image_name = [self.coco.loadImgs(img_id)[0]['file_name'] for img_id in self.images_id]

        self.texts  = clip.clip.tokenize(self.captions)
        self.img_length = len(set(self.images_id))
        self.txt_length = len(self.captions)
        logger.info('%d images have been loaded.', self.img_length)
        logger.info("%s set initialization completed!", subset)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[idx]))) # Image from PIL module
        text = self.texts[idx]
        img_id = self.images_id[idx]

        return image, text, img_id