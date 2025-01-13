import os

import torch.utils.data
from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, data_dir, crop="random_resize", image_size=256):
        """ crop: ['random', 'random_resize', 'center', 'none']
        default crop size: 256x256
        """
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

        crop_name = crop.lower()
        if crop_name == 'random':
            Crop = transforms.RandomCrop
        elif crop_name == 'random_resize':
            Crop = transforms.RandomResizedCrop
        elif crop_name == 'center':
            Crop = transforms.CenterCrop
        elif crop_name == 'none':
            Crop = None
        else:
            raise NotImplementedError("Invalid crop type!")

        self.transform = transforms.Compose([
            Crop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]) if Crop else \
            transforms.Compose([
                transforms.ToTensor(),
            ])

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        image = self.transform(image)
        image = image * 2.0 - 1.0
        return image

    def __len__(self):
        return len(self.image_path)


class Datasets_WithName(Dataset):
    def __init__(self, data_dir, crop="random_resize", image_size=256):
        """ crop: ['random', 'random_resize', 'center', 'none']
        default crop size: 256x256
        """
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

        crop_name = crop.lower()
        if crop_name == 'random':
            Crop = transforms.RandomCrop
        elif crop_name == 'random_resize':
            Crop = transforms.RandomResizedCrop
        elif crop_name == 'center':
            Crop = transforms.CenterCrop
        elif crop_name == 'none':
            Crop = None
        else:
            raise NotImplementedError("Invalid crop type!")

        self.transform = transforms.Compose([
            Crop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]) if Crop else \
            transforms.Compose([
                transforms.ToTensor(),
            ])

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        image = self.transform(image)
        image = image * 2.0 - 1.0
        return image, image_ori

    def __len__(self):
        return len(self.image_path)


if __name__ == "__main__":
    print("hello")

