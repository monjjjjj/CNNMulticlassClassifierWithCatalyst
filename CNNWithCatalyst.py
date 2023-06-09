from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, HorizontalFlip,
    ToFloat, VerticalFlip
)
import catalyst
from catalyst.dl import SupervisedRunner
from catalyst.utils import get_one_hot
from catalyst.dl.callbacks.metrics import AccuracyCallback, AUCCallback
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from glob import glob
import torchvision
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from skimage.io import imread
import torch.nn.functional as F
from scipy.special import softmax
import random
from torchvision import datasets

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# check device
SEED = 42
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(SEED)

# create dataset
data_dir = '/home/chloe/Siting/ALASKA2'
folder_names = ['JMiPOD/', 'JUNIWARD/', 'UERD/']
class_names = ['Normal', 'JMiPOD_75', 'JMiPOD_90', 'JMiPOD_95', 
               'JUNIWARD_75', 'JUNIWARD_90', 'JUNIWARD_95',
                'UERD_75', 'UERD_90', 'UERD_95']
class_labels = { name: i for i, name in enumerate(class_names)}

#train_df = pd.read_csv('../input/alaska2trainvalsplit/alaska2_train_df.csv')
#val_df = pd.read_csv('../input/alaska2trainvalsplit/alaska2_val_df.csv')

class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None, test = False):
        self.data = df
        self.augment = augmentations
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.test:
            fn = self.data.loc[idx][0]
        else:
            fn, label = self.data.loc[idx]
        im = imread(fn)
        if self.augment:
            im = self.augment(image=im)
        if self.test:
            item = {'features': im['image']}
        else:
            item = {'features': im['image'], 'targets':label, 'bool_targets': get_one_hot(label, 10)}

        return item


AUGMENTATIONS_TRAIN = Compose([
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    ToFloat(max_value=255),
    ToTensorV2(),
], p=1)


AUGMENTATIONS_TEST = Compose([
    ToFloat(max_value=255),
    ToTensorV2()
], p=1)


batch_size = 32
num_workers = 8

train_data = datasets.ImageFolder(data_dir, augmentations = AUGMENTATIONS_TRAIN)
#train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
#valid_dataset = Alaska2Dataset(val_df.sample(5000).reset_index(drop=True), augmentations=AUGMENTATIONS_TEST)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)

# valid_loader = torch.utils.data.DataLoader(valid_dataset,
#                                            batch_size=batch_size*2,
#                                            num_workers=num_workers,
#                                            shuffle=False)

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_name('efficientnet-b0')
        self.dense_output = nn.Linear(1280, num_classes)

    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)

loaders = {
    "train": train_loader,
    #"valid": valid_loader
}

model = Net(num_classes=len(class_labels))
model.load_state_dict(torch.load('checkpoints/best-checkpoint-042epoch_3_c.bin'))

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
criterion = torch.nn.CrossEntropyLoss()
callbacks = [
    AccuracyCallback(),
    AUCCallback(input_key='bool_targets', num_classes = 1) # I was too lazy to implement weighted AUC. It is strongly correlated with regular AUC. But it should be easy to make your own catalyst "meter" for weighted AUC 
]

runner = SupervisedRunner()

test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({'ImageFileName': list(
    test_filenames)}, columns=['ImageFileName'])

batch_size = 16
num_workers = 4
test_dataset = datasets.ImageFolder(test_df, augmentations = AUGMENTATIONS_TEST, test=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False,
                                          drop_last=False)

model.cuda()
preds = []
for outputs in tqdm(runner.predict_loader(loader=test_loader, model=model)):
    preds.append(softmax(outputs))

preds = np.array(preds)

test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
test_df['Label'] = 1-preds[:, 0]

test_df = test_df.drop('ImageFileName', axis=1)
test_df.to_csv('submission.csv', index=False)
print(test_df.head())