import csv
import json
import os
import pickle
import random

import albumentations as A
import cv2
import lightning as L
import numpy as np
import scipy
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
from torch import nn
import torchvision
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torchmetrics
import torch.nn.functional as F


# ============================== 1 Classifier model ============================
class CarsDetectionDataset(Dataset):
    def __init__(self, features, target):
        super(CarsDetectionDataset).__init__()
        
        self.features = features
        self.target = target
        
    def __getitem__(self, index):
        return self.features[index], self.target[index]
    
    def __len__(self):
        return len(self.features)
    
class LightningCarsDetection(L.LightningModule):

    def __init__(self, *, lr=1e-3, simple_classifier=None, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.simple_classifier = simple_classifier
        self.model = self.get_model()
        self.loss_fn = nn.CrossEntropyLoss()
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def get_model(self):
        if self.simple_classifier is not None:
            cls_model = self.simple_classifier.model
            cls_model.eval()
            detection_model = nn.Sequential(OrderedDict([
                    (f'conv1', cls_model.conv1),
                    (f'act1', cls_model.act1),
                    (f'conv2', cls_model.conv2),
                    (f'act2', cls_model.act2),
                    (f'max_pool_1', cls_model.max_pool_1),
                    (f'conv3', cls_model.conv3),
                    (f'batch_norm1', cls_model.batch_norm1),
                    (f'act3', cls_model.act3),
                    (f'dropout', cls_model.dropout),
                    (f'conv4', cls_model.conv4),
                    (f'batch_norm2', cls_model.batch_norm2),
                    (f'act4', cls_model.act4),
                    (f'max_pool_2', cls_model.max_pool_2),
                    (f'conv5', nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(10, 25))),
                    (f'act_final', cls_model.act_final),
                    # (f'dropout', nn.Dropout(p=0.1)),
                    (f'conv6', nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)),
                    ('softmax', nn.Softmax(dim=1)),
                ]))
            detection_model.eval()
            
            with torch.no_grad():
                layer1_dict = cls_model.lin1.state_dict().copy()
                layer1_dict['weight'] = layer1_dict['weight'].reshape(detection_model.conv5.state_dict()['weight'].shape)
                layer1_dict['bias'] = layer1_dict['bias'].reshape(detection_model.conv5.state_dict()['bias'].shape)
                detection_model.conv5.load_state_dict(layer1_dict)
                
                layer2_dict = cls_model.lin2.state_dict().copy()
                layer2_dict['weight'] = layer2_dict['weight'].reshape(detection_model.conv6.state_dict()['weight'].shape)
                layer2_dict['bias'] = layer2_dict['bias'].reshape(detection_model.conv6.state_dict()['bias'].shape)
                detection_model.conv6.load_state_dict(layer2_dict)
                
            return detection_model
        
        model = nn.Sequential(OrderedDict([
            (f'conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same")),
            (f'act1', nn.ReLU()),
            (f'conv2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")),
            (f'act2', nn.ReLU()),
            (f'max_pool_1', nn.MaxPool2d(kernel_size=2)),
            (f'conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")),
            (f'batch_norm1', nn.BatchNorm2d(64)),
            (f'act3', nn.ReLU()),
            (f'dropout', nn.Dropout(p=0.2)),
            (f'conv4', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")),
            (f'batch_norm2', nn.BatchNorm2d(64)),
            (f'act4', nn.ReLU()),
            (f'max_pool_2', nn.MaxPool2d(kernel_size=2)),
            (f'flat', nn.Flatten()),
            (f'lin1', nn.Linear(in_features=16000, out_features=256)),
            (f'act_final', nn.ReLU()),
            # (f'dropout', nn.Dropout(p=0.1)),
            (f'lin2', nn.Linear(in_features=256, out_features=2)),
        ]))
        
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.01)

        # lr_scheduler = {
        #     "scheduler": scheduler,
        #     "interval": "epoch",
        #     "frequency": 1,
        #     "monitor": "train_loss",
        # }
        # return [optimizer], [lr_scheduler]
        return optimizer

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")

    def _step(self, batch, kind):
        imgs, target = batch
        pred = self.model(imgs)

        loss = self.loss_fn(pred, target)
        accs = torch.sum(pred.argmax(axis=1) == target) / target.shape[0]

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
            self.logger.experiment.add_scalars("accuracy", {f"{kind}_accs": accs}, self.global_step)
        return loss
    
    def forward(self, imgs):
        return self.model(imgs)
    
def train_model(
    model,
    experiment_path,
    dl_train,
    dl_val,
    max_epochs=10,
    fast_train=False,
    ckpt_path=None,
    monitor="val_loss",
    **trainer_kwargs,
):
    callbacks = [
        L.pytorch.callbacks.LearningRateMonitor(),
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
            max_epochs=max_epochs,
            default_root_dir=experiment_path,
            enable_checkpointing=False,
            accelerator="cpu",
            logger=False,
        )
    trainer.fit(model, dl_train, dl_val, ckpt_path=ckpt_path)

def get_cls_model(input_shape=(1, 40, 100)):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    classification_model = LightningCarsDetection(lr=1e-3)
    return classification_model
    # your code here /\


def fit_cls_model(X, y, fast_train=True, X_val=None, y_val=None):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model()
    
    batch_size = 32
    num_workers = 0 if fast_train else 2
    
    ds_train = CarsDetectionDataset(X, y)
    dataloader_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    
    if fast_train is False:
        ds_val = CarsDetectionDataset(X_val, y_val)
        dataloader_val = DataLoader(
            dataset=ds_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        train_model(
            model,
            "./simple_classifier",
            dataloader_train,
            dataloader_val,
            accelerator="auto",
            max_epochs=5,
        )
        
        torch.save(model.state_dict(), './classifier_model.pt')
    else:
        train_model(
            model,
            "./simple_classifier",
            dataloader_train,
            None,
            max_epochs=5,
            fast_train=True,
        )
    
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    model = LightningCarsDetection(lr=1e-3, simple_classifier=cls_model)
    detection_model = model.model
    
    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    result_detections = {}
    coef_resizing = 4
    
    detection_model.eval()
    with torch.no_grad():
        for filename, image in dictionary_of_images.items():
            result_detections[filename] = []
            padded_img = np.zeros((1, 1, 220, 370), dtype=np.float32)
            padded_img[:, :, :image.shape[0], :image.shape[1]] = image
            
            detection_preds = detection_model(torch.tensor(padded_img, dtype=torch.float32)).detach().cpu().numpy()[0]
            cars_preds = detection_preds[1]

            for h in range(0, min(image.shape[0] // coef_resizing, cars_preds.shape[0])):
                for w in range(0, min(image.shape[1] // coef_resizing, cars_preds.shape[1])):
                    if cars_preds[h][w] >= 0.0:
                        result_detections[filename].append([
                            coef_resizing * h,
                            coef_resizing * w,
                            40,
                            100,
                            cars_preds[h][w]
                        ])
            
    return result_detections
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    x1 = max(first_bbox[0], second_bbox[0])
    y1 = max(first_bbox[1], second_bbox[1])
    x2 = min(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
    y2 = min(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])

    height = max(y2 - y1, 0)
    width = max(x2 - x1, 0)
    area_of_intersection = height * width
    
    area_of_union = first_bbox[2] * first_bbox[3] + second_bbox[2] * second_bbox[3] - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    return iou
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    tp = []
    fp = []
    
    tp_fn_count = 0
    for _, gt_bbox in gt_bboxes.items():
        tp_fn_count += len(gt_bbox)
    
    for filename, bbox_preds in pred_bboxes.items():
        pred_bboxes = sorted(bbox_preds, key=lambda x: x[-1], reverse=True)
        cur_gt_bboxes = gt_bboxes[filename]

        for pred in pred_bboxes:
            max_iou = 0
            idx_max = -1
            for i, box in enumerate(cur_gt_bboxes):
                cur_iou = calc_iou(pred[:-1], box)
                if cur_iou > max_iou:
                    max_iou = cur_iou
                    idx_max = i
                    
            if idx_max != -1 and max_iou >= 0.5:
                tp.append(pred[-1])
                cur_gt_bboxes.remove(cur_gt_bboxes[idx_max])
            else:
                fp.append(pred[-1])
    
    tp_and_fp = tp + fp            
    tp_and_fp = np.array(sorted(tp_and_fp))
    tp = np.array(sorted(tp))
    
    precision_recall_all = []
    # precision_recall_all.append((0, 1, 1))
    precision_recall_all.append((len(tp) / tp_fn_count, len(tp) / len(tp_and_fp), 0))
    for prediction in tp_and_fp:
        cur_conf = prediction
        greater_in_tp_fp = np.sum(tp_and_fp >= cur_conf)
        greater_in_tp = np.sum(tp >= cur_conf)
        
        precision = greater_in_tp / greater_in_tp_fp
        recall = greater_in_tp / tp_fn_count
        
        precision_recall_all.append((recall, precision, cur_conf))
    precision_recall_all.append((0, 1, 1))
    # precision_recall_all.append((len(tp) / tp_fn_count, len(tp) / len(tp_and_fp), 0))
    
    auc = 0
    for i, metric in enumerate(precision_recall_all):
        if i != len(precision_recall_all) - 1:
            auc += abs(precision_recall_all[i + 1][0] - metric[0]) * (precision_recall_all[i + 1][1] + metric[1]) / 2
    
    return auc      
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.3):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    detections_new = {}
    for filename, detection in detections_dictionary.items():
        detections = sorted(detection, key=lambda x: x[-1], reverse=True)
        
        detections_new[filename] = []
        for det in detections:
            if len(detections_new[filename]) == 0:
                detections_new[filename].append(det)
                continue
            
            find_nearest = False
            for det_new in detections_new[filename]:
                if calc_iou(det_new[:-1], det[:-1]) > iou_thr:
                    find_nearest = True
                    break
                    
            if not find_nearest:
                detections_new[filename].append(det)
                
    return detections_new
    # your code here /\
