import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr

import cv2
import os
import pandas as pd
import albumentations as A
import random

from collections import defaultdict
from tqdm import tqdm


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


MODEL_INPUT_SIZE = (100, 100)

resize_compression = A.Compose([
    A.Resize(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1], interpolation=cv2.INTER_AREA),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

resize_expansion = A.Compose([
    A.Resize(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1], interpolation=cv2.INTER_LINEAR),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

resize_compression_wo_points = A.Compose([
    A.Resize(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1], interpolation=cv2.INTER_AREA),
])

resize_expansion_wo_points = A.Compose([
    A.Resize(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1], interpolation=cv2.INTER_LINEAR),
])

train_transformations = A.Compose([
    A.OneOf([
        A.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=0.5),
        A.ColorJitter(
            brightness=0.3,
            contrast=0.7,
            saturation=0.3,
            p=0.6
        ),
        # A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.6),
        # A.GaussNoise(var_limit=(10, 50), mean=0, per_channel=True, p=0.5),
        # A.InvertImg(p=0.3),
        # A.RGBShift(p=0.6),
        A.ToGray(p=0.3),
        A.GaussianBlur(p=0.4)
    ]),
    #A.HorizontalFlip(p=0.2),
    #A.RandomRotate90(p=0.5),
    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=[0.53631788, 0.43026288, 0.37494377], std=[0.23770446, 0.21814688, 0.20839559], always_apply=True),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

val_transformations = A.Compose([
    A.Normalize(mean=[0.53631788, 0.43026288, 0.37494377], std=[0.23770446, 0.21814688, 0.20839559], always_apply=True),
])

test_transformations = A.Compose([
    A.Normalize(mean=[0.53631788, 0.43026288, 0.37494377], std=[0.23770446, 0.21814688, 0.20839559], always_apply=True),
])


class ImageKeypointsDataset(Dataset):
    def __init__(self, imgs_path, imgs_list, gt_keypoints_dict, train_fraction=0.8, train=True):
        super(ImageKeypointsDataset).__init__()
        
        self.imgs_path = imgs_path
        self.imgs_list = imgs_list
        self.gt_keypoints_dict = gt_keypoints_dict
        self.train_fraction = train_fraction
        self.train = train
        
        self.train_transformations = train_transformations
        self.val_transformations = val_transformations
        self.resize_compression = resize_compression
        self.resize_expansion = resize_expansion
        
        rng = random.Random()
        split = int(self.train_fraction * len(self.imgs_list))
        rng.shuffle(self.imgs_list)
        
        if self.train == True:
            self.imgs_list = self.imgs_list[:split]
        else:
            self.imgs_list = self.imgs_list[split:]
        
        

    def __getitem__(self, index):
        image = np.asarray(PIL.Image.open(os.path.join(self.imgs_path, self.imgs_list[index])))
        if len(image.shape) < 3:
            image = np.stack([image] * 3, axis=-1)
        keypoints = np.array(self.gt_keypoints_dict[self.imgs_list[index]])
        
        if image.shape[0] < MODEL_INPUT_SIZE[0] or image.shape[1] < MODEL_INPUT_SIZE[1]:
            resized = self.resize_expansion(image=image, keypoints=keypoints.reshape(-1, 2))
        else:
            resized = self.resize_compression(image=image, keypoints=keypoints.reshape(-1, 2))
        resized_img = resized['image']
        resized_keypoints = np.array(resized['keypoints'], dtype=np.float32).ravel()
            
        if self.train == True:
            transformed = self.train_transformations(image=resized_img, keypoints=resized_keypoints.reshape(-1, 2))
            transformed_img = transformed['image']
            transformed_keypoints = np.array(transformed['keypoints'], dtype=np.float32).ravel()
            if (transformed_keypoints < 0.0).sum() > 0:
                transformed = self.val_transformations(image=resized_img)
                transformed_img = transformed['image']
                transformed_keypoints = resized_keypoints
        else:
            transformed = self.val_transformations(image=resized_img)
            transformed_img = transformed['image']
            transformed_keypoints = resized_keypoints
        
        return torch.from_numpy(transformed_img), torch.from_numpy(transformed_keypoints)
    
    def __len__(self):
        return len(self.imgs_list)
    
    
def get_dataloaders(img_folder, all_names, train_coords, train_fraction, batch_size, num_workers):
    ds_train = ImageKeypointsDataset(
        img_folder,
        all_names,
        train_coords,
        train=True,
        train_fraction=train_fraction,
    )

    ds_val = ImageKeypointsDataset(
        img_folder,
        all_names,
        train_coords,
        train=False,
        train_fraction=train_fraction,
    )
    
    dataloader_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    
    return dataloader_train, dataloader_val


# realisation light version of ResNet with modifications of original model from https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = x + identity
        x = self.relu(x)

        return x
    
class ResNet50(nn.Module):
    def __init__(self, num_classes, layer_list=[1,1,1,1], num_channels=3):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, layer_list[0], planes=64)
        self.layer2 = self._make_layer(Bottleneck, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(Bottleneck, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(Bottleneck, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_1 = nn.Linear(512, 256)
        self.batch_norm = nn.BatchNorm1d(256)
        self.fc_2 = nn.Linear(256, 28)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_2(self.relu(self.batch_norm(self.fc_1(x))))

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        layers = []

        layers.append(ResBlock(self.in_channels, planes, stride=stride))
        self.in_channels = planes

        return nn.Sequential(*layers)
    
    
# most useful functions from SHAD course ML part 2 (seminars and homework templates)   

def train(model, optimizer, loader, criterion, fast_train):
    model.train()
    losses_tr = []
    num_batch = 0
    for img_batch, coords_batch in tqdm(loader):
        img_batch = img_batch.to(device)
        target = coords_batch.to(device)
        
        optimizer.zero_grad()
        
        pred = model(img_batch)
        loss = criterion(pred, target)
        
        loss.backward()
        optimizer.step()
        losses_tr.append(loss.item())
        
        num_batch = num_batch + 1
        if fast_train == True and num_batch == 2:
            break
    
    return model, optimizer, np.mean(losses_tr)


def val(model, loader, criterion, fast_train, metric_names=None):
    model.eval()
    losses_val = []
    num_samples = 0
    num_batch = 0
    # i = 0
    if metric_names:
        metrics = {name: 0 for name in metric_names}
    # fig, ax = plt.subplots(h, w, figsize=(20, 15))
    with torch.no_grad():
        for img_batch, coords_batch in tqdm(loader):
            img_batch = img_batch.to(device)
            target = coords_batch.to(device)
            
            pred = model(img_batch)
            loss = criterion(pred, target)

            losses_val.append(loss.item())
            
            if metric_names is not None:
                num_samples += len(pred)
                if 'mse' in metrics:
                    metrics['mse'] += (torch.sum(torch.mean(torch.square(pred - target), dim=-1))).detach().cpu().numpy()
                    
            # if i < 10:
            #     keypoints = pred[0, :].detach().cpu().numpy()
            #     gt = target[0, :].detach().cpu().numpy()
            #     image = img_batch[0, :].detach().cpu()
            #     plt.scatter(keypoints[0::2], keypoints[1::2], c='r')
            #     plt.scatter(gt[0::2], gt[1::2], c='b')
            #     plt.imshow(de_normalize(image))
            #     plt.subplot(h, w, i+1)
            #     i = i + 1
            
            num_batch = num_batch + 1
            if fast_train == True and num_batch == 2:
                break

        if metric_names is not None:
            for name in metrics:
                metrics[name] = metrics[name] / num_samples
                print(name, " : ", metrics[name])
    # plt.show()
    return np.mean(losses_val), metrics if metric_names else None


def save_checkpoint(path, model, optimizer, epoch, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
    
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss

# from IPython.display import clear_output

def learning_loop(model, optimizer, train_loader, val_loader, criterion, fast_train, scheduler=None, epochs=10, val_every=1, draw_every=1, save_every=1, metric_names=None, check_path=None):
    losses = {'train': [], 'val': []}
    if metric_names:
        metrics = {name: [] for name in metric_names}

    for epoch in range(1, epochs+1):
        print(f'#{epoch}/{epochs}:')
        model, optimizer, loss = train(model, optimizer, train_loader, criterion, fast_train)
        losses['train'].append(loss)

        if not (epoch % val_every):
            loss, metrics_ = val(model, val_loader, criterion, fast_train, metric_names)
            losses['val'].append(loss)
            if metric_names:
                for name in metrics_:
                    metrics[name].append(metrics_[name])
            if scheduler:
                try:
                    scheduler.step()
                except:
                    scheduler.step(loss)
                    
        if not (epoch % save_every):
            if check_path is not None:
                save_checkpoint(check_path, model, optimizer, epoch, loss)

        if not (epoch % draw_every):
            # clear_output(True)
            ww = 2 if metric_names else 1
            fig, ax = plt.subplots(1, ww, figsize=(20, 10))
            fig.suptitle(f'#{epoch}/{epochs}:')

            plt.subplot(1, ww, 1)
            plt.title('losses')
            plt.plot(losses['train'], 'r.-', label='train')
            plt.plot(losses['val'], 'g.-', label='val')
            plt.legend()
            
            if metric_names:
                plt.subplot(1, ww, 2)
                plt.title('additional metrics')
                for name in metric_names:
                    plt.plot(metrics[name], '.-', label=name)
                plt.legend()
            
            plt.show()
    
    return model, optimizer, losses, metrics if metric_names else None


def detect(path_to_model, path_to_imgs, device=torch.device('cpu')):
    all_img_names = list(filter(lambda name: (name.split('.')[1] == "jpg"), os.listdir(path_to_imgs)))
    
    pred_coords = {}
    
    checkpoint = torch.load(path_to_model, map_location=torch.device('cpu'))
    model_res = ResNet50(num_classes=28)
    model_res.load_state_dict(checkpoint['model_state_dict'])
    model_res = model_res.to(device)

    model_res.eval()
    with torch.no_grad():
        for idx in range(len(all_img_names)):
            img_name = all_img_names[idx]
            image = np.asarray(PIL.Image.open(os.path.join(path_to_imgs, img_name)))
            
            if len(image.shape) < 3:
                image = np.stack([image] * 3, axis=-1)
            if image.shape[0] < MODEL_INPUT_SIZE[0] or image.shape[1] < MODEL_INPUT_SIZE[1]:
                resized = resize_expansion_wo_points(image=image)
            else:
                resized = resize_compression_wo_points(image=image)

            normalized_img = test_transformations(image=resized['image'])['image']
            normalized_img = torch.from_numpy(normalized_img)[None, :].to(device)

            pred = model_res(normalized_img).detach().cpu().numpy()

            pred_keypoints = pred[0, :]

            if MODEL_INPUT_SIZE[0] < image.shape[0] and MODEL_INPUT_SIZE[1] < image.shape[1]:
                expansion_to_orig = A.Compose([
                    A.Resize(height=image.shape[0], width=image.shape[1], interpolation=cv2.INTER_LINEAR),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

                keypoints_orig = np.array(expansion_to_orig(image=resized['image'], keypoints=pred_keypoints.reshape(-1, 2))['keypoints'], dtype=np.float32).ravel()

            else:
                compression_to_orig = A.Compose([
                    A.Resize(height=image.shape[0], width=image.shape[1], interpolation=cv2.INTER_AREA),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

                keypoints_orig = np.array(compression_to_orig(image=resized['image'], keypoints=pred_keypoints.reshape(-1, 2))['keypoints'], dtype=np.float32).ravel()

            pred_coords[img_name] = keypoints_orig.tolist()

    return pred_coords


def train_detector(gt_coords, path_to_imgs, fast_train):
    all_names = list(filter(lambda name: (name.split('.')[1] == "jpg"), os.listdir(path_to_imgs)))
    
    batch_size = 64
    if fast_train == True:
        batch_size = 2
    dataloader_train, dataloader_val = get_dataloaders(path_to_imgs, all_names, gt_coords, train_fraction=0.95, batch_size=batch_size, num_workers=0)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if fast_train == True:
        device = torch.device('cpu')
        
    model = ResNet50(num_classes=28).to(device)

    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=0.001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        verbose=True,
        eta_min=0.00001,
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, threshold=0.1, min_lr=0.00001)
    criterion = nn.MSELoss()
    
    num_epoch = 50
    if fast_train == True:
        num_epoch = 1
        scheduler = None
    
    model, optimizer, losses, metrics = learning_loop(
        model=model,
        optimizer=optimizer,
        train_loader=dataloader_train,
        val_loader=dataloader_val,
        criterion=criterion,
        scheduler=scheduler,
        fast_train=fast_train,
        epochs=num_epoch,
        val_every=1,
        draw_every=2,
        save_every=2,
        check_path="./checkpoint_model.pt",
        metric_names=['mse'],
    )
    
    if fast_train == False:
        save_checkpoint("./facepoints_model.pt", model, optimizer, epoch=num_epoch, loss=losses['val'][-1])