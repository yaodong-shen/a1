import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from .randaugment import RandomAugment
from .utils_data import generate_uniform_cv_candidate_labels
import random

mean, std = {}, {}
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]


class stl10_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels, transform=None):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.transform = transform

        crop_size = 96
        crop_ratio = 0.875
        if self.transform is None:
            self.weak_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(crop_size),
                    transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)),
                                      padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean['stl10'], std=std['stl10'])])
            self.strong_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(crop_size),
                    transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)),
                                          padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(3, 5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean['stl10'], std=std['stl10'])])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        img = self.images[index]
        img = np.transpose(img, (1, 2, 0))

        if self.transform is None:
            each_image_w = self.weak_transform(img)
            each_image_s = self.strong_transform(img)
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]

            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            each_label = self.given_label_matrix[index]
            each_image = self.transform(img)
            each_true_label = self.true_labels[index]
            return each_image, each_label, each_true_label


def load_stl10(partial_rate, batch_size):
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean['stl10'], std=std['stl10'])])

    temp_train_unlabeled = dsets.STL10(root='../data/stl10', split='train+unlabeled', download=True, transform=transforms.ToTensor())

    print(torch.Tensor(temp_train_unlabeled.data).size(), torch.Tensor(temp_train_unlabeled.labels).size())
    train_data = temp_train_unlabeled.data[:5000]
    labels = torch.Tensor(temp_train_unlabeled.labels[:5000]).long()
    train_unlabeled = temp_train_unlabeled.data[5000:]
    labels_unlabeled = torch.Tensor(temp_train_unlabeled.labels[5000:]).long()
    # get original data and labels

    test_dataset = dsets.STL10(root='../data/stl10', split='test', download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size * 4, shuffle=False,
                                              num_workers=4)

    partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    train_label_cnt = torch.unique(labels, sorted=True, return_counts=True)[-1]

    partialY_unlabeled = torch.ones(len(train_unlabeled), partialY.shape[1])
    partialY = torch.cat((partialY, partialY_unlabeled))
    data = np.concatenate((train_data, train_unlabeled), axis=0)
    labels = torch.cat((labels, labels_unlabeled))
    print('Average candidate num: ', partialY.sum(1).mean())

    # generate partial labels

    partial_matrix_dataset = stl10_Augmentention(data, partialY.float(), labels.float())
    # generate partial label dataset

    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              num_workers=4,
                                                              pin_memory=True,
                                                              drop_last=True)
    return partial_matrix_train_loader, partialY, test_loader, train_label_cnt

