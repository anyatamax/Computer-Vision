import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
import albumentations as A
import albumentations.pytorch.transforms
import lightning as L

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr
import torchvision
import torchmetrics

import cv2
import os
import pandas as pd
import random
from collections import OrderedDict
from sklearn.model_selection import train_test_split


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def get_imgs_names(path_to_img):
    return list(filter(lambda name: (name.split('.')[1] == "jpg" or name.split('.')[1] == "png"), os.listdir(path_to_img)))

def convert_lables_to_dict(path):
    lables = {}
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            lables[row[0]] = int(row[1])
            
    return lables

MODEL_INPUT_SIZE = (288, 288)

train_transformations = A.Compose([
    A.Resize(height=300, width=300, interpolation=cv2.INTER_LINEAR),
    A.CenterCrop(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1]),
    A.OneOf([
        A.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=0.5),
        A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.6),
        A.GaussNoise(var_limit=(10, 50), mean=0, per_channel=True, p=0.5),
        A.GaussianBlur(p=0.4),
        A.RandomToneCurve(p=0.5)
    ]),
    A.OneOf([
        A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=0.4),
        A.Rotate(limit=45, p=0.5),
    ]),
    A.Normalize(mean=[0.49050629, 0.51356942, 0.46767225], std=[0.18116629, 0.18073399, 0.19205182], always_apply=True),
])

val_transformations = A.Compose([
    A.Resize(height=300, width=300, interpolation=cv2.INTER_LINEAR),
    A.CenterCrop(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1]),
    A.Normalize(mean=[0.49050629, 0.51356942, 0.46767225], std=[0.18116629, 0.18073399, 0.19205182], always_apply=True),
])


class ImageBirdsDataset(Dataset):
    def __init__(self, imgs_path, imgs_list, lables_gt, test_fraction=0.1, train=True):
        super(ImageBirdsDataset).__init__()
        
        self.imgs_path = imgs_path
        self.imgs_list = imgs_list
        self.lables_gt = lables_gt
        self.test_fraction = test_fraction
        self.train = train
        
        self.train_transformations = train_transformations
        self.val_transformations = val_transformations
        
        imgs_train, imgs_val = train_test_split(self.imgs_list, test_size=self.test_fraction, stratify=list(self.lables_gt.values()), random_state=42)
        
        if self.train == True:
            self.imgs_list = imgs_train
        else:
            self.imgs_list = imgs_val
        
    def __getitem__(self, index):
        image = np.asarray(PIL.Image.open(os.path.join(self.imgs_path, self.imgs_list[index])))
        if len(image.shape) < 3:
            image = np.stack([image] * 3, axis=-1)
        lable_gt = self.lables_gt[self.imgs_list[index]]
            
        if self.train == True:
            transformed_img = self.train_transformations(image=image)['image']
        else:
            transformed_img = self.val_transformations(image=image)['image']
        
        return torch.from_numpy(transformed_img).permute(2, 0, 1), lable_gt
    
    def __len__(self):
        return len(self.imgs_list)
    
    
def get_dataloaders(img_folder, imgs_names, gt_lables, test_fraction, batch_size, num_workers):
    ds_train = ImageBirdsDataset(
        img_folder,
        imgs_names,
        gt_lables,
        train=True,
        test_fraction=test_fraction,
    )

    ds_val = ImageBirdsDataset(
        img_folder,
        imgs_names,
        gt_lables,
        train=False,
        test_fraction=test_fraction,
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



def get_frozen_mobilenet_v2(num_classes, transfer=True):
    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if transfer else None
    model = torchvision.models.mobilenet_v2(weights=weights)

    model.classifier = nn.Sequential(OrderedDict([
        (f'dropout1', nn.Dropout(p=0.3, inplace=False)),
        (f'batchnorm1', nn.BatchNorm1d(1280)),
        (f'lin1', nn.Linear(in_features=1280, out_features=512, bias=True)),
        (f'batchnorm2', nn.BatchNorm1d(512)),
        (f'lin2', nn.Linear(in_features=512, out_features=num_classes, bias=True)),
    ]))
    
    model_features = model.features

    for child in list(model_features.children()):
        for param in child.parameters():
            param.requires_grad = False

    return model

def get_frozen_efficientnet_b2(num_classes, transfer=True):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT if transfer else None
    model = torchvision.models.efficientnet_b2(weights=weights)

    model.classifier = nn.Sequential(OrderedDict([
        (f'dropout1', nn.Dropout(p=0.4, inplace=False)),
        (f'batchnorm1', nn.BatchNorm1d(1408)),
        (f'lin1', nn.Linear(in_features=1408, out_features=512, bias=True)),
        (f'batchnorm2', nn.BatchNorm1d(512)),
        (f'dropout2', nn.Dropout(p=0.3, inplace=False)),
        (f'lin2', nn.Linear(in_features=512, out_features=num_classes, bias=True)),
    ]))
    
    model_features = model.features

    for child in list(model_features.children()):
        for param in child.parameters():
            param.requires_grad = False

    return model

def get_frozen_last_efficientnet_b2(num_classes, transfer=True):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT if transfer else None
    model = torchvision.models.efficientnet_b2(weights=weights)

    model.classifier = nn.Sequential(OrderedDict([
        (f'dropout1', nn.Dropout(p=0.4, inplace=False)),
        (f'batchnorm1', nn.BatchNorm1d(1408)),
        (f'lin1', nn.Linear(in_features=1408, out_features=512, bias=True)),
        (f'act', nn.SiLU(inplace=True)),
        (f'batchnorm2', nn.BatchNorm1d(512)),
        (f'dropout2', nn.Dropout(p=0.3, inplace=False)),
        (f'lin2', nn.Linear(in_features=512, out_features=num_classes, bias=True)),
    ]))
    
    model_features = model.features
    model_avgpool = model.avgpool

    for child in list(model.features.children())[:-1]:
        for param in child.parameters():
            param.requires_grad = False

    return model

def warmup_then_cosine_annealing_lr(
    optimizer,
    start_factor,
    T_max,
    warmup_duration,
):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_duration,
    )
    cos_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=0.00001,
    )
    warmup_then_cos_anneal = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup, cos_annealing],
        milestones=[warmup_duration],
    )
    return warmup_then_cos_anneal


class LightningBirdsClassifier(L.LightningModule):
    num_classes = 50

    def __init__(self, *, transfer=True, lr=1e-4, steps_per_epoch=35, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.transfer = transfer
        self.steps_per_epoch = steps_per_epoch
        self.model = self.get_model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
        )

    def get_model(self):
        return get_frozen_last_efficientnet_b2(
            self.num_classes,
            self.transfer,
        )

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

        steps_per_epoch = self.steps_per_epoch
        warmup_duration = 0.4 * steps_per_epoch

        scheduler = warmup_then_cosine_annealing_lr(
            optimizer,
            start_factor=0.05,
            T_max=30,
            warmup_duration=warmup_duration,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")

    def _step(self, batch, kind):
        imgs, target = batch
        pred = self.model(imgs)

        loss = self.loss_fn(pred, target)
        accs = self.accuracy(pred.argmax(axis=-1), target)

        return self._log_metrics(loss, accs, kind)

    def _log_metrics(self, loss, accs, kind):
        metrics = {}
        if loss is not None:
            metrics[f"{kind}_loss"] = loss
        if accs is not None:
            metrics[f"{kind}_accs"] = accs
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        if self.logger is not None:
            self.logger.experiment.add_scalars("losses", {f"{kind}_loss": loss}, self.global_step)
            self.logger.experiment.add_scalars("accuracy", {f"{kind}_loss": accs}, self.global_step)
        return loss
    
    def forward(self, imgs):
        return self.model(imgs)
    
    
def train_model(
    model,
    experiment_path,
    dl_train,
    dl_valid,
    max_epochs=100,
    fast_train=False,
    ckpt_path=None,
    **trainer_kwargs,
):
    callbacks = [
        # L.pytorch.callbacks.TQDMProgressBar(),
        L.pytorch.callbacks.LearningRateMonitor(),
        L.pytorch.callbacks.ModelCheckpoint(
            filename="{epoch}-{val_accs:.3f}",
            monitor="val_accs",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_accs",
            mode="max",
            patience=10,
            verbose=True,
        )
    ]
    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        default_root_dir=experiment_path,
        **trainer_kwargs,
    )
    if fast_train == True:
        trainer = L.Trainer(
            callbacks=None,
            max_epochs=1,
            default_root_dir=experiment_path,
            max_steps=2,
            num_sanity_val_steps=1,
            log_every_n_steps=50,
            enable_checkpointing=False,
            accelerator="cpu",
            logger=False,
        )
    trainer.fit(model, dl_train, dl_valid, ckpt_path=ckpt_path)
    
    
def train_classifier(train_gt, train_img_dir, fast_train=False):
    img_names = get_imgs_names(train_img_dir)
    
    batch_size = 64
    if fast_train == True:
        batch_size = 2
    dataloader_train, dataloader_val = get_dataloaders(train_img_dir, img_names, train_gt, test_fraction=0.15, batch_size=batch_size, num_workers=0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if fast_train == True:
        device = 'cpu'
        
    transfer = True
    if fast_train == True:
        transfer = False
    efficientnet_b2 = LightningBirdsClassifier(
                        transfer=transfer,
                        lr=1e-4,
                        steps_per_epoch=len(dataloader_train))
    train_model(
        efficientnet_b2,
        "./birds",
        dataloader_train,
        dataloader_val,
        accelerator=device,
        max_epochs=25,
        fast_train=fast_train,
    )
    
    return efficientnet_b2
    
    
def classify(model_path, test_img_dir):
    img_names = get_imgs_names(test_img_dir)
    
    pred_lables = {}
    
    model_res = LightningBirdsClassifier.load_from_checkpoint(
        model_path,
        map_location=torch.device('cpu'),
        lr=1e-4,
        transfer=False)

    model_res.eval()
    with torch.no_grad():
        for idx in range(len(img_names)):
            img_name = img_names[idx]
            image = np.asarray(PIL.Image.open(os.path.join(test_img_dir, img_name)))
            
            if len(image.shape) < 3:
                image = np.stack([image] * 3, axis=-1)
            transformed_img = val_transformations(image=image)['image']
            transformed_img = torch.from_numpy(transformed_img).permute(2, 0, 1)[None, :]

            pred = model_res(transformed_img).numpy()
            pred_lable = pred[0, :]

            pred_lables[img_name] = pred_lable.argmax()

    return pred_lables
