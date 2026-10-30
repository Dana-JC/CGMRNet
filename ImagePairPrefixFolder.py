import os
import bisect
import threading
import torch
import numpy as np
import numpy.random as random
import cv2
from PIL import Image
from torch.utils.data import Dataset
from folder_loader import FolderLoader
import torchvision.transforms as transforms
from utils import batch_edge_compute

def pil_loader(img_path):
    return Image.open(img_path).convert("RGB")


class ImagePairPrefixFolder(Dataset):
    def __init__(self, input_folder, gt_folder, max_img_size=0, size_unit=1, force_rgb=False, crop_size=None,
                 use_clahe=False, use_gamma=False, gamma_val=0.6, use_bilateral=False, use_flip=False
                 ):
        super(ImagePairPrefixFolder, self).__init__()

        self.gt_loader = FolderLoader(gt_folder)
        self.gt_map = dict()
        for idx, img_name in enumerate(self.gt_loader.img_names):
            self.gt_map[os.path.splitext(img_name)[0].split('_')[0]] = idx

        self.input_loader = FolderLoader(input_folder)
        assert all([os.path.splitext(x)[0].split('_')[0] in self.gt_map for x in self.input_loader.img_names]), \
                'cannot find corresponding gt names'

        self.input_folder = input_folder
        self.gt_folder = gt_folder
        self.max_img_size = max_img_size
        self.size_unit = size_unit
        self.force_rgb = force_rgb
        self.crop_size = crop_size if crop_size is not None else 256
        
        self.use_clahe = use_clahe
        self.use_gamma = use_gamma
        self.gamma_val = gamma_val
        self.use_bilateral = use_bilateral
        self.use_flip = use_flip


    def __getitem__(self, index):
        input_name, input_img = self.input_loader[index]
        input_basename = os.path.splitext(input_name)[0].split('_')[0]
        gt_idx = self.gt_map[input_basename]

        gt_name, gt_img = self.gt_loader[gt_idx]
        if self.force_rgb:
            input_img = input_img.convert('RGB')
            gt_img = gt_img.convert('RGB')
        im_w, im_h = input_img.size
        gt_w, gt_h = gt_img.size
        assert im_w==gt_w and im_h==gt_h, 'input image and gt image size not match'

        im_w, im_h = input_img.size
        if 0 < self.max_img_size < max(im_w, im_h):
            if im_w < im_h:
                out_h = int(self.max_img_size) // self.size_unit * self.size_unit
                out_w = int(im_w / im_h * out_h) // self.size_unit * self.size_unit
            else:
                out_w = int(self.max_img_size) // self.size_unit * self.size_unit
                out_h = int(im_h / im_w * out_w) // self.size_unit * self.size_unit
        else:
            out_w = im_w // self.size_unit * self.size_unit
            out_h = im_h // self.size_unit * self.size_unit

        if im_w != out_w or im_h != out_h:
            input_img = input_img.resize((out_w, out_h), Image.BILINEAR)
            gt_img = gt_img.resize((out_w, out_h), Image.BILINEAR)

        if self.use_flip:
            if random.rand() > 0.5:
                input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
                gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.rand() > 0.5:
                input_img = input_img.transpose(Image.FLIP_TOP_BOTTOM)
                gt_img = gt_img.transpose(Image.FLIP_TOP_BOTTOM)

        im_w, im_h = input_img.size

        if self.use_clahe or self.use_gamma or self.use_bilateral:
            img_np = np.array(input_img)
            is_color = (len(img_np.shape) == 3 and img_np.shape[2] == 3)

            if self.use_bilateral:
                img_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)

            if self.use_clahe:
                if is_color:
                    ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    y_eq = clahe.apply(y)
                    img_np = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2RGB)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_np = clahe.apply(img_np.squeeze())

            if self.use_gamma:
                img_float = img_np.astype(np.float32) / 255.0
                img_gamma = np.power(img_float, self.gamma_val) * 255.0
                img_np = np.clip(img_gamma, 0, 255).astype(np.uint8)

            input_img = Image.fromarray(img_np)


        input_img = np.array(input_img).astype(np.float32)
        gt_img = np.array(gt_img).astype(np.float32)
        if len(input_img.shape) == 2:
            input_img = input_img[:, :, np.newaxis]
        if len(gt_img.shape) == 2:
            gt_img = gt_img[:, :, np.newaxis]
        return {'input_img': input_img, 'gt_img': gt_img,  'input_h': im_h, "input_w": im_w,    "crop_size": self.crop_size}

    def get_input_info(self, index):
        image_name = os.path.splitext(self.input_loader.img_names[index])[0]
        return self.input_loader, image_name

    def __len__(self):
        return len(self.input_loader)


def var_custom_collate(batch):

    crop_size = batch[0]['crop_size']

    inc = 1 if len(batch[0]['input_img'].shape)==2 else batch[0]['input_img'].shape[2]

    batch_input_images = torch.Tensor(
        len(batch),
        inc,
        crop_size,
        crop_size
    )

    batch_gt_images = torch.Tensor(
        len(batch),
        inc,
        crop_size,
        crop_size
    )


    for idx,item in enumerate(batch):

        h,w,_ = item['input_img'].shape


        # 如果图片小于256，先resize
        if h < crop_size or w < crop_size:

            input_img = cv2.resize(
                item['input_img'],
                (max(w,crop_size), max(h,crop_size))
            )

            gt_img = cv2.resize(
                item['gt_img'],
                (max(w,crop_size), max(h,crop_size))
            )

        else:
            input_img = item['input_img']
            gt_img = item['gt_img']


        h,w,_ = input_img.shape


        # 随机crop
        off_y = random.randint(
            0,
            h-crop_size+1
        )

        off_x = random.randint(
            0,
            w-crop_size+1
        )


        crop_input_img = input_img[
            off_y:off_y+crop_size,
            off_x:off_x+crop_size,
            :
        ]

        crop_gt_img = gt_img[
            off_y:off_y+crop_size,
            off_x:off_x+crop_size,
            :
        ]


        batch_input_images[idx] = torch.from_numpy(
            crop_input_img.transpose((2,0,1))
        ) - 128


        batch_gt_images[idx] = torch.from_numpy(
            crop_gt_img.transpose((2,0,1))
        )


    batch_input_edges = batch_edge_compute(
        batch_input_images
    ) - 128


    return batch_input_images,batch_input_edges,batch_gt_images
    
def test_collate(batch):

    input_imgs=[]
    gt_imgs=[]

    for item in batch:

        input_img=item['input_img']
        gt_img=item['gt_img']


        input_imgs.append(
            torch.from_numpy(
                input_img.transpose(2,0,1)
            )-128
        )

        gt_imgs.append(
            torch.from_numpy(
                gt_img.transpose(2,0,1)
            )
        )

    return (
        torch.stack(input_imgs),
        torch.zeros_like(torch.stack(input_imgs)),
        torch.stack(gt_imgs)
    )