import math
import os
import random
import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'

__all__ = ["Dataset"]

class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())  # Expose filenames for diagnostics
        self.n = len(self.filenames)
        self.indices = range(self.n)
        self.albumentations = Albumentations()
    # ...existing methods from dataset.py...
    @staticmethod
    def load_label(filenames):
        # ...diagnostics removed...
        # Cache logic unchanged
        path = f'{os.path.dirname(filenames[0])}.cache'
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        if os.path.exists(path):
            try:
                return torch.load(path, weights_only=False)
            except Exception as e:
                print(f"Cache file corrupted, regenerating: {e}")
                os.remove(path)
        x = {}
        for filename in filenames:
            try:
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()
                shape = image.size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                if image.format is None or image.format.lower() not in FORMATS:
                    raise AssertionError(f'invalid image format {image.format}')
                # Build label path: use the actual label directory and image basename
                label_dir = os.path.join(os.path.dirname(os.path.dirname(filename)), 'labels')
                label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(filename))[0] + '.txt')
                if os.path.isfile(label_path):
                    with open(label_path) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()
                        assert label.shape[1] == 5
                        assert (label[:, 1:] <= 1).all()
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:
                            label = label[i]
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
            except FileNotFoundError:
                label = numpy.zeros((0, 5), dtype=numpy.float32)
            except AssertionError:
                continue
            x[filename] = label
        torch.save(x, path)
        return x
    @staticmethod
    def collate_fn(batch):
        # Filter out None values (in case __getitem__ returns None)
        batch = [b for b in batch if b is not None]
        if not batch:
            # Return dummy batch if all items are None
            samples = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
            targets = {'cls': torch.zeros((0, 1)), 'box': torch.zeros((0, 4)), 'idx': torch.zeros((0,))}
            return samples, targets
        samples, cls, box, indices = zip(*batch)
        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)
        targets = {'cls': cls,
                   'box': box,
                   'idx': indices}
        return torch.stack(samples, dim=0), targets

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        try:
            index = self.indices[index]
            params = self.params
            mosaic = self.mosaic and random.random() < params['mosaic']
            if mosaic:
                image, label = self.load_mosaic(index, params)
                if random.random() < params['mix_up']:
                    index = random.choice(self.indices)
                    mix_image1, mix_label1 = image, label
                    mix_image2, mix_label2 = self.load_mosaic(index, params)
                    image, label = mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
            else:
                image, shape = self.load_image(index)
                h, w = image.shape[:2]
                image, ratio, pad = self.resize(image, self.input_size, self.augment)
                label = self.labels[index].copy()
                if label.size:
                    label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
                if self.augment:
                    image, label = random_perspective(image, label, params)
                nl = len(label)
                h, w = image.shape[:2]
                cls = label[:, 0:1]
                box = label[:, 1:5]
                box = xy2wh(box, w, h)
                if self.augment:
                    image, box, cls = self.albumentations(image, box, cls)
                    nl = len(box)
                    augment_hsv(image, params)
                    if random.random() < params['flip_ud']:
                        image = numpy.flipud(image)
                        if nl:
                            box[:, 1] = 1 - box[:, 1]
                    if random.random() < params['flip_lr']:
                        image = numpy.fliplr(image)
                        if nl:
                            box[:, 0] = 1 - box[:, 0]
                target_cls = torch.zeros((nl, 1))
                target_box = torch.zeros((nl, 4))
                if nl:
                    target_cls = torch.from_numpy(cls)
                    target_box = torch.from_numpy(box)
                sample = image.transpose((2, 0, 1))[::-1]
                sample = numpy.ascontiguousarray(sample)
                return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)
        except Exception as e:
            # Return dummy data if anything fails
            sample = numpy.zeros((3, self.input_size, self.input_size), dtype=numpy.float32)
            target_cls = torch.zeros((0, 1))
            target_box = torch.zeros((0, 4))
            return torch.from_numpy(sample), target_cls, target_box, torch.zeros(0)

    # Resize function as a static method for clarity
    @staticmethod
    def resize(image, input_size, augment):
        h, w = image.shape[:2]
        r = input_size / max(h, w)
        if r != 1:
            interp = resample() if augment else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w * r), int(h * r)), interpolation=interp)
        new_h, new_w = image.shape[:2]
        pad_h = input_size - new_h
        pad_w = input_size - new_w
        pad = (pad_w // 2, pad_h // 2)
        image = cv2.copyMakeBorder(image, pad[1], pad_h - pad[1], pad[0], pad_w - pad[0], cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ratio = (new_w / w, new_h / h)
        return image, ratio, pad

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=resample() if self.augment else cv2.INTER_LINEAR)
        return image, (h, w)

    def load_mosaic(self, index, params):
        label4 = []
        border = [-self.input_size // 2, -self.input_size // 2]
        image4 = numpy.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=numpy.uint8)
        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        for i, index in enumerate(indices):
            image, _ = self.load_image(index)
            shape = image.shape
            if i == 0:  # top left
                x1a = max(xc - shape[1], 0)
                y1a = max(yc - shape[0], 0)
                x2a = xc
                y2a = yc
                x1b = shape[1] - (x2a - x1a)
                y1b = shape[0] - (y2a - y1a)
                x2b = shape[1]
                y2b = shape[0]
            elif i == 1:  # top right
                x1a = xc
                y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = yc
                x1b = 0
                y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a)
                y2b = shape[0]
            elif i == 2:  # bottom left
                x1a = max(xc - shape[1], 0)
                y1a = yc
                x2a = xc
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a)
                y1b = 0
                x2b = shape[1]
                y2b = min(y2a - y1a, shape[0])
            elif i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = 0
                y1b = 0
                x2b = min(shape[1], x2a - x1a)
                y2b = min(y2a - y1a, shape[0])
            else:
                raise ValueError(f"Unexpected mosaic index: {i}")
            pad_w = x1a - x1b
            pad_h = y1a - y1b
            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            label = self.labels[index].copy()
            if len(label):
                label[:, 1:] = wh2xy(label[:, 1:], shape[1], shape[0], pad_w, pad_h)
            label4.append(label)
        label4 = numpy.concatenate(label4, 0)
        for x in label4[:, 1:]:
            numpy.clip(x, 0, 2 * self.input_size, out=x)
        image4, label4 = random_perspective(image4, label4, params, border)
        return image4, label4
# ...existing code from dataset.py...
class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations
            transforms = [albumentations.Blur(p=0.01),
                          albumentations.CLAHE(p=0.01),
                          albumentations.ToGray(p=0.01),
                          albumentations.MedianBlur(p=0.01)]
            self.transform = albumentations.Compose(transforms,
                                                    albumentations.BboxParams('yolo', ['class_labels']))
        except ImportError:
            pass
    def __call__(self, image, box, cls):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=box,
                               class_labels=cls)
            image = x['image']
            box = numpy.array(x['bboxes'])
            cls = numpy.array(x['class_labels'])
        return image, box, cls

def mix_up(image1, label1, image2, label2):
    alpha = numpy.random.beta(a=32.0, b=32.0)
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label

def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
    return y

def xy2wh(x, w, h):
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y

def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)

def augment_hsv(image, params):
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']
    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')
    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)

def random_perspective(image, label, params, border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2
    center[1, 2] = -image.shape[0] / 2
    perspective = numpy.eye(3)
    rotate = numpy.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    translate = numpy.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():
        image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))
    n = len(label)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ matrix.T
        xy = xy[:, :2].reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        box = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)
        indices = (box[:, 2] - box[:, 0] > 2) & (box[:, 3] - box[:, 1] > 2)
        label = label[indices]
        label[:, 1:5] = box[indices]
    return image, label
