import json
import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize


def data_processing(data):
    images = []
    labels = []
    for sample in data:
        filepath = sample[0]
        label = sample[1]
        image = Image.open(filepath)
        x = TF.to_tensor(image)[0, :, :]
        x.unsqueeze_(0)
        x = Resize([256, 256])(x)
        images.append(x)
        if label == 'bad':
            labels.append(0)
        if label == 'average':
            labels.append(1)
        if label == 'good':
            labels.append(2)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def predict_preprocessing(filepath):
    image = Image.open(filepath)
    x = TF.to_tensor(image)[0, :, :]
    x.unsqueeze_(0).unsqueeze_(0)
    x = Resize([256, 256])(x)
    return x


class TrainDataset(data.Dataset):
    def __init__(self):
        with open(train_data_path) as fp:
            self.dict_data = json.load(fp)

    def __getitem__(self, index):
        x = self.dict_data['filepath'][index]
        y = self.dict_data['label'][index]
        return [x, y]

    def __len__(self):
        return len(self.dict_data['filepath'])


class TestDataset(data.Dataset):
    def __init__(self):
        with open(test_data_path) as fp:
            self.dict_data = json.load(fp)

    def __getitem__(self, index):
        x = self.dict_data['filepath'][index]
        y = self.dict_data['label'][index]
        return [x, y]

    def __len__(self):
        return len(self.dict_data['filepath'])


train_data_path = 'train_data.json'
test_data_path = 'test_data.json'
