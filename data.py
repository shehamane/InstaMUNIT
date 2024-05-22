"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import random

import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import os.path

def rgb_loader(path):
    return Image.open(path).convert('RGB')

def greyscale_loader(path):
    return Image.open(path).convert('L')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist



class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=rgb_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=rgb_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class InstaDataset(data.Dataset):
    def __init__(self, imgs_root, labels_root, return_paths=False,
                 img_loader=rgb_loader, label_loader=greyscale_loader):
        imgs = sorted(make_dataset(imgs_root))
        labels = sorted(make_dataset(labels_root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + imgs_root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.imgs_root = imgs_root
        self.labels_root = labels_root
        self.imgs = imgs
        self.labels = labels
        self.return_paths = return_paths
        self.img_loader = img_loader
        self.label_loader = label_loader

    def transform(self, image, label):
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image = TF.to_pil_image(image)
        label = TF.to_pil_image(label, 'L')

        i, j, h, w = T.RandomCrop.get_params(image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        image = TF.resize(image, 100)
        label = TF.resize(label, 100)

        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        return image, label

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = self.img_loader(img_path)
        label_path = self.labels[index]
        label = self.label_loader(label_path)
        if self.transform is not None:
            img, label = self.transform(img, label)
        if self.return_paths:
            return img, label, img_path
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)


class ImageFolder(data.Dataset):

    def __init__(self, root, return_paths=False,
                 loader=rgb_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.return_paths = return_paths
        self.loader = loader


    def transform(self, image):
        image = TF.to_tensor(image)

        image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image = TF.to_pil_image(image)

        i, j, h, w = T.RandomCrop.get_params(image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)

        image = TF.resize(image, 100)

        if random.random() > 0.5:
            image = TF.hflip(image)

        image = TF.to_tensor(image)

        return image

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
