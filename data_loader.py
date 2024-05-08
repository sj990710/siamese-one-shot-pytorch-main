import os
import random
from random import Random

import Augmentor
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as dset, transforms

def get_train_validation_loader(data_dir, batch_size, num_train, augment, way, trials, shuffle, seed, num_workers,
                                pin_memory):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')

    train_transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8444], std=[0.5329])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8444], std=[0.5329])
    ])

    train_dataset = dset.ImageFolder(train_dir, transform=train_transform)
    train_dataset = OmniglotTrain(train_dataset, num_train, augment)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_memory)

    val_dataset = dset.ImageFolder(val_dir, transform=val_transform)
    val_dataset = Omniglotvalid(val_dataset, trials, way, seed)
    val_loader = DataLoader(val_dataset, batch_size=way, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=pin_memory)

    return train_loader, val_loader

# def get_test_loader(data_dir_1, data_dir_2, way, trials, seed, num_workers, pin_memory):
#     test_dir_1 = os.path.join(data_dir_1, 'test')
#     test_dir_2 = os.path.join(data_dir_2, 'test_query')
#
#     test_transform = transforms.Compose([
#         transforms.Resize((105, 105)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.8444], std=[0.5329])
#     ])
#
#     test_dataset_1 = dset.ImageFolder(test_dir_1, transform=test_transform)
#     test_dataset_1 = OmniglotTest(test_dataset_1, trials=trials, way=way, seed=seed)
#     test_loader_1 = DataLoader(test_dataset_1, batch_size=way, shuffle=False, num_workers=num_workers,
#                              pin_memory=pin_memory)
#
#     return test_loader_1
def get_test_loader(data_dir, way, trials, seed, num_workers, pin_memory):
    test_dir = os.path.join(data_dir, 'test')
    test_transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8444], std=[0.5329])
    ])
    test_dataset = dset.ImageFolder(test_dir,transform=test_transform)
    test_dataset = OmniglotTest(test_dataset, trials=trials, way=way, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=way, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    return test_loader
def get_visual_loader(data_dir, way, trials, seed, num_workers, pin_memory):
    test_dir_1 = os.path.join(data_dir, 'test')
    test_dir_2 = os.path.join(data_dir, 'test_query')

    # 이미지 변환 설정
    visual_transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8444], std=[0.5329])
    ])

    # test_dataset_1에 이미지 변환 적용
    test_dataset_1 = dset.ImageFolder(test_dir_1, transform=visual_transform)
    test_dataset_1 = OmniglotTest_sample(test_dataset_1, trials, way, seed)
    test_loader_1 = DataLoader(test_dataset_1, batch_size=way, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # test_dataset_2에 이미지 변환 적용
    test_dataset_2 = dset.ImageFolder(test_dir_2, transform=visual_transform)
    test_dataset_2 = OmniglotTest_query(test_dataset_2, trials=1, way=1, seed=seed)  # trials와 way를 1로 설정하여 단일 이미지 로드
    test_loader_2 = DataLoader(test_dataset_2, batch_size=1, shuffle=False, num_workers=num_workers,
                               pin_memory=pin_memory)

    return test_loader_1, test_loader_2


# adapted from https://github.com/fangpin/siamese-network
class OmniglotTrain(Dataset):

    def __init__(self, dataset, num_train, augment):
        self.dataset = dataset
        self.num_train = num_train
        self.augment = augment

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        if index % 2 == 1:
            label = 1.0
            idx = random.randint(0, len(self.dataset.classes) - 1)
            image_list = [x for x in self.dataset.imgs if x[1] == idx]
            image1 = random.choice(image_list)
            image2 = random.choice(image_list)
            while image1[0] == image2[0]:
                image2 = random.choice(image_list)
        else:
            label = 0.0
            image1 = random.choice(self.dataset.imgs)
            image2 = random.choice(self.dataset.imgs)
            while image1 == image2:
                image2 = random.choice(self.dataset.imgs)

        trans = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8444], std=[0.5329])
        ])

        image1 = Image.open(image1[0]).convert('L')
        image2 = Image.open(image2[0]).convert('L')
        image1 = trans(image1)
        image2 = trans(image2)
        label = torch.tensor(label, dtype=torch.float32)

        return image1, image2, label

# class Omniglotvalid(Dataset):
#     def __init__(self, dataset, trials, way, seed=0):
#         self.dataset = dataset
#         self.trials = trials
#         self.way = way
#         self.seed = seed
#         self.image1 = None
#
#     def __len__(self):
#         return self.trials * self.way
#
#     def __getitem__(self, index):
#         rand = Random(self.seed + index)
#         if index % self.way == 0:  # 새로운 'way'마다 anchor 이미지를 선택
#             idx = rand.randint(0, len(self.dataset.classes) - 1)
#             image_list = [x for x in self.dataset.imgs if x[1] == idx]
#             self.image1 = rand.choice(image_list)  # anchor 이미지 선택
#             image2 = rand.choice(image_list)
#             while self.image1[0] == image2[0]:  # 다른 이미지 선택
#                 image2 = rand.choice(image_list)
#             label = 1.0  # 같은 클래스
#         else:
#             image2 = random.choice(self.dataset.imgs)
#             while self.image1[1] == image2[1]:  # 다른 클래스 이미지 선택
#                 image2 = random.choice(self.dataset.imgs)
#             label = 0.0  # 다른 클래스
#
#         # 이미지 변환을 적용하기 전에 레이블 정보를 추출
#         image2_label = image2[1]  # 이미지 변환 전에 image2의 레이블을 추출
#
#         # 이미지 변환 적용
#         trans = transforms.Compose([
#             transforms.Resize((105, 105)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.8444], std=[0.5329])
#         ])
#
#         image1 = Image.open(self.image1[0]).convert('L')
#         image2 = Image.open(image2[0]).convert('L')
#         image1 = trans(image1)
#         image2 = trans(image2)
#
#         anchor_label = self.image1[1]  # anchor 이미지의 클래스 인덱스
#
#         return image1, image2, torch.tensor(label, dtype=torch.float32), torch.tensor(anchor_label,dtype=torch.int64), torch.tensor(image2_label, dtype=torch.int64)
#

class Omniglotvalid(Dataset):
    def __init__(self, dataset, trials, way, seed=0):
        self.dataset = dataset
        self.trials = trials
        self.way = way
        self.seed = seed
        self.image1 = None
        self.initialize_anchor_image()  # 앵커 이미지 초기화

    def initialize_anchor_image(self):
        rand = Random(self.seed)
        idx = rand.randint(0, len(self.dataset.classes) - 1)
        image_list = [x for x in self.dataset.imgs if x[1] == idx]
        self.image1 = rand.choice(image_list)

    def __len__(self):
        return self.trials * self.way

    def __getitem__(self, index):
        rand = Random(self.seed + index)
        if index % self.way == 0:
            idx = rand.randint(0, len(self.dataset.classes) - 1)
            image_list = [x for x in self.dataset.imgs if x[1] == idx]
            self.image1 = rand.choice(image_list)
            image2 = rand.choice(image_list)
            while self.image1[0] == image2[0]:  # 같은 클래스에서 다른 이미지 선택
                image2 = rand.choice(image_list)
            label = 1.0  # 같은 클래스
        else:
            # 'self.image1'가 초기화되었는지 확인
            if self.image1 is None:
                raise ValueError("Anchor image is not initialized.")
            image2 = rand.choice(self.dataset.imgs)
            while self.image1[1] == image2[1]:  # 다른 클래스 이미지 선택
                image2 = rand.choice(self.dataset.imgs)
            label = 0.0  # 다른 클래스

        image2_label = image2[1]

        trans = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8444], std=[0.5329])
        ])

        image1 = Image.open(self.image1[0]).convert('L')
        image2 = Image.open(image2[0]).convert('L')
        image1 = trans(image1)
        image2 = trans(image2)

        anchor_label = self.image1[1]  # anchor 이미지의 클래스 인덱스

        return image1, image2, torch.tensor(label, dtype=torch.float32), torch.tensor(anchor_label,
                                                                                      dtype=torch.int64), torch.tensor(
            image2_label, dtype=torch.int64)

# class OmniglotTest:
#     def __init__(self, dataset, trials, way, seed=0):
#         self.dataset = dataset
#         self.trials = trials
#         self.way = way
#         self.seed = seed
#         self.image1 = None
#         self.mean = 0.8444
#         self.std = 0.5329

#     def __len__(self):
#         return self.trials * self.way

#     def __getitem__(self, index):
#         rand = Random(self.seed + index)
#         # get image pair from same class
#         if index % self.way == 0:
#             label = 1.0
#             idx = rand.randint(0, len(self.dataset.classes) - 1)
#             image_list = [x for x in self.dataset.imgs if x[1] == idx]
#             self.image1 = rand.choice(image_list)
#             image2 = rand.choice(image_list)
#             while self.image1[0] == image2[0]:
#                 image2 = rand.choice(image_list)

#         # get image pair from different class
#         else:
#             label = 0.0
#             image2 = random.choice(self.dataset.imgs)
#             while self.image1[1] == image2[1]:
#                 image2 = random.choice(self.dataset.imgs)

#         trans = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=self.mean, std=self.std)
#         ])

#         image1 = Image.open(self.image1[0]).convert('L')
#         image2 = Image.open(image2[0]).convert('L')
#         image1 = trans(image1)
#         image2 = trans(image2)

#         return image1, image2, label
class OmniglotTest(Dataset):
  def __init__(self, dataset, trials, way, seed=0, transform=None):
      self.dataset = dataset
      self.trials = trials
      self.way = way
      self.seed = seed
      self.image1 = None
      self.transform = transform or transforms.Compose([
          transforms.Resize((105, 105)),  # 이미지 리사이즈 추가
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.8444], std=[0.5329])
    ])
  def __len__(self):
      return self.trials * self.way

  def __getitem__(self, index):
      rand = Random(self.seed + index)
      # 같은 클래스의 이미지 쌍을 선택
      if index % self.way == 0:
          label = 1.0
          idx = rand.randint(0, len(self.dataset.classes) - 1)
          image_list = [x for x in self.dataset.imgs if x[1] == idx]
          self.image1 = rand.choice(image_list)
          image2 = rand.choice(image_list)
          while self.image1[0] == image2[0]:
              image2 = rand.choice(image_list)

      # 다른 클래스의 이미지 쌍을 선택
      # 다른 클래스의 이미지 쌍을 선택
      else:
          label = 0.0
          image2 = rand.choice(self.dataset.imgs)  # 수정됨
          while self.image1[1] == image2[1]:
              image2 = rand.choice(self.dataset.imgs)  # 수정됨


      image1 = Image.open(self.image1[0]).convert('L')
      image2 = Image.open(image2[0]).convert('L')
      
      if self.transform:
          image1 = self.transform(image1)
          image2 = self.transform(image2)

      return image1, image2, label
      
class OmniglotTest_sample(Dataset):
    def __init__(self, dataset, trials, way, seed=0):
        self.dataset = dataset
        self.trials = trials
        self.way = way
        self.seed = seed
        self.classes = [d for d in os.listdir(dataset.root) if os.path.isdir(os.path.join(dataset.root, d))]
        self.selected_images = self.select_images()

    def select_images(self):
        random.seed(self.seed)
        selected_images = []
        for _ in range(self.trials):
            way_images = []
            for _ in range(self.way):
                cls = random.choice(self.classes)
                class_dir = os.path.join(self.dataset.root, cls)
                images = os.listdir(class_dir)
                selected_image = random.choice(images)
                way_images.append((os.path.join(class_dir, selected_image), cls))
            selected_images.append(way_images)
        return selected_images

    def __len__(self):
        return self.trials

    def __getitem__(self, index):
        way_images = self.selected_images[index]

        # 이미지 변환 적용
        trans = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8444], std=[0.5329])
        ])

        images, labels = [], []
        for img_path, cls in way_images:
            image = Image.open(img_path).convert('L')
            image = trans(image)
            images.append(image)
            labels.append(self.classes.index(cls))

        return torch.stack(images), torch.tensor(labels, dtype=torch.int64)


# class OmniglotTest_query(Dataset):
#     def __init__(self, dataset, trials, way, seed=0):
#         self.dataset = dataset
#         self.trials = trials
#         self.way = way
#         self.seed = seed
#         self.classes = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
#         self.selected_images = self.select_images()
#
#     def select_images(self):
#         random.seed(self.seed)
#         selected_class = random.choice(self.classes)
#         selected_class_dir = os.path.join(self.dataset, selected_class)
#         all_items = os.listdir(selected_class_dir)
#         images = [item for item in all_items if os.path.isfile(os.path.join(selected_class_dir, item))]
#
#         if images:
#             selected_image = random.choice(images)
#             image_path = os.path.join(selected_class_dir, selected_image)
#             return [(image_path, self.classes.index(selected_class))]
#         else:
#             raise Exception(f"No images found in directory {selected_class_dir}")
#
#     def __len__(self):
#         return self.trials * self.way
#
#     def __getitem__(self, index):
#         # 매번 호출 시 무작위 클래스 선택
#         selected_class = random.choice(self.classes)
#         selected_class_dir = os.path.join(self.dataset, selected_class)
#         all_items = os.listdir(selected_class_dir)
#         images = [item for item in all_items if os.path.isfile(os.path.join(selected_class_dir, item))]
#
#         # 선택된 클래스에서 무작위로 이미지 선택
#         selected_image = random.choice(images)
#         image_path = os.path.join(selected_class_dir, selected_image)
#         label = self.classes.index(selected_class)
#
#         # 이미지 변환 적용
#         trans = transforms.Compose([
#             transforms.Resize((105, 105)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.8444], std=[0.5329])
#         ])
#
#         image = Image.open(image_path).convert('L')
#         image = trans(image)
#
#         return image, torch.tensor(label, dtype=torch.int64)

class OmniglotTest_query(Dataset):
    def __init__(self, dataset, trials, way, seed=0):
        self.dataset = dataset
        self.trials = trials
        self.way = way
        self.seed = seed
        self.classes = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset.root, d))]
        self.image_path, self.selected_class_index = self.select_image()  # 단일 이미지 선택으로 변경

    def select_image(self):
        random.seed(self.seed)
        selected_class = random.choice(self.classes)
        selected_class_dir = os.path.join(self.dataset.root, selected_class)
        all_items = os.listdir(selected_class_dir)
        images = [item for item in all_items if os.path.isfile(os.path.join(selected_class_dir, item))]

        if images:
            selected_image = random.choice(images)
            image_path = os.path.join(selected_class_dir, selected_image)
            return image_path, self.classes.index(selected_class)
        else:
            raise Exception(f"No images found in directory {selected_class_dir}")

    def __len__(self):
        return self.trials * self.way

    def __getitem__(self, index):
        # 이미지 변환 적용
        trans = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8444], std=[0.5329])
        ])

        image = Image.open(self.image_path).convert('L')
        image = trans(image)
        return image, torch.tensor(self.selected_class_index, dtype=torch.int64)
