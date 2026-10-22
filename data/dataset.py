import torch
import torchvision.transforms.functional as TF
from PIL import Image

from data.augment import color_jitter, random_scale_translate


class DetectionDataset(torch.utils.data.Dataset):
    """samples: list of {"image_path": str, "boxes": [[x1,y1,x2,y2], ...], "labels": [int, ...]}.

    Fully dataset-agnostic: knows nothing about COCO, Roboflow, or class names -- it
    just turns a plain sample list into training tensors. Images are resized (stretched,
    not letterboxed) to img_size x img_size, and boxes are scaled to match.

    augment: training-split only. Random scale + translate (boxes moved with the image,
    see data/augment.py) and colour jitter -- never flips.
    """

    def __init__(self, samples, img_size=320, augment=False, scale_range=(0.7, 1.3), translate=0.1):
        self.samples = samples
        self.img_size = img_size
        self.augment = augment
        self.scale_range = scale_range
        self.translate = translate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.img_size, self.img_size))

        sx = self.img_size / orig_w
        sy = self.img_size / orig_h

        if sample["boxes"]:
            boxes_tensor = torch.tensor(sample["boxes"], dtype=torch.float32)
            boxes_tensor[:, [0, 2]] *= sx
            boxes_tensor[:, [1, 3]] *= sy
            labels_tensor = torch.tensor(sample["labels"], dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        if self.augment:
            image, boxes_tensor, labels_tensor = random_scale_translate(
                image, boxes_tensor, labels_tensor, self.scale_range, self.translate
            )
            image = color_jitter(image)

        image_tensor = TF.to_tensor(image)  # (3, img_size, img_size), values in 0-1
        return image_tensor, boxes_tensor, labels_tensor


def collate_fn(batch):
    """Images stack normally; boxes/labels stay ragged lists since each image has a
    different box count."""
    images, boxes, labels = zip(*batch)
    return torch.stack(images, dim=0), list(boxes), list(labels)
