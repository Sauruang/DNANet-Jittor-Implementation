from PIL import Image, ImageOps, ImageFilter
import platform, os
from jittor.dataset import Dataset
import random
import numpy as np
import jittor as jt
from jittor import nn, init
from datetime import datetime
import argparse

class TrainSetLoader(Dataset):
    """红外小目标分割训练数据集加载器"""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png'):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = os.path.join(dataset_dir, 'masks')
        self.images = os.path.join(dataset_dir, 'images')
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _sync_transform(self, img, mask):
        # 随机镜像翻转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # 随机缩放（短边）
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # 填充裁剪
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # 随机裁剪到指定大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # 高斯模糊（如PSP中使用）
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # 最终变换
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):
        img_id     = self._items[idx]             
        img_path   = os.path.join(self.images, img_id + self.suffix)
        label_path = os.path.join(self.masks, img_id + self.suffix)

        img = Image.open(img_path).convert('RGB')        
        mask = Image.open(label_path)

        # 同步变换
        img, mask = self._sync_transform(img, mask)

        # 通用调整大小、归一化和转换为张量
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0

        return img, jt.array(mask) #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """红外小目标分割测试数据集加载器"""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = os.path.join(dataset_dir, 'masks')
        self.images    = os.path.join(dataset_dir, 'images')
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # 最终变换
        img, mask = np.array(img), np.array(mask, dtype=np.float32) 
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  
        img_path   = os.path.join(self.images, img_id + self.suffix)
        label_path = os.path.join(self.masks, img_id + self.suffix)
        img  = Image.open(img_path).convert('RGB') 
        mask = Image.open(label_path)
        # 同步变换
        img, mask = self._testval_sync_transform(img, mask)

        # 通用调整大小、归一化和转换为张量
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        return img, jt.array(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)


def weights_init_xavier(m):
    """Xavier权重初始化函数"""
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        jt.init.xavier_normal_(m.weight)


class AverageMeter(object):
    """计算并存储平均值和当前值的工具类"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计值"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """更新统计值"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(net, filepath):
    """保存模型检查点"""
    jt.save(net, filepath)


def weights_init_kaiming(m):
    """Kaiming权重初始化函数"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        jt.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        jt.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)

