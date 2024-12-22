# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor

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
from albumentations.pytorch import ToTensorV2
from PIL import Image
import PIL
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torchmetrics
import torch.nn.functional as F

# !Этих импортов достаточно для решения данного задания


CLASSES_CNT = 205
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_SIZE = (224, 224)


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.

    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """

    def __init__(
        self,
        root_folders: typing.List[str],
        path_to_classes_json: str,
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        ### список пар (путь до картинки, индекс класса)
        self.samples = []
        for forlder in root_folders:
            for img_type in os.listdir(forlder):
                path_to_imgs_cur_type = os.path.join(forlder, img_type)
                if not os.path.isdir(path_to_imgs_cur_type):
                    continue
                for img in os.listdir(path_to_imgs_cur_type):
                    path_to_img = os.path.join(path_to_imgs_cur_type, img)
                    self.samples.append((path_to_img, self.class_to_idx[img_type]))
                    
        ### cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.classes_to_samples = {i: [] for i in range(len(self.classes))}
        for i in range(len(self.samples)):
            self.classes_to_samples[self.samples[i][1]].append(i)
        ### аугментации + нормализация + ToTensorV2
        self.transform = A.Compose([
            A.Resize(height=250, width=250, interpolation=cv2.INTER_LINEAR), # 232 with simple
            A.RandomCrop(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1]),
            A.OneOf([
                # A.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=0.3),
                # A.RandomToneCurve(p=0.4)
            ]),
            A.ColorJitter(brightness=(1, 1), contrast=(1, 1), hue=(0,0), saturation=(0.5, 1.0), p=1.0), # only with syntetic
            A.OneOf([
                # A.HorizontalFlip(p=0.5),
                # A.Rotate(limit=45, p=0.5),
            ]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1),
            ToTensorV2(),
        ])

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        path_to_img, class_idx = self.samples[index]
        image = self.transform(image=np.asarray(Image.open(path_to_img).convert('RGB')))['image']
        
        return image, path_to_img, class_idx
        

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """
        ### словарь, class_to_idx['название класса'] = индекс
        ### массив, classes[индекс] = 'название класса'
        with open(path_to_classes_json) as file:
            classes_json = json.load(file)
            
            class_to_idx = {img_type: info["id"] for img_type, info in classes_json.items()}
            classes = [""] * len(class_to_idx)
            for img_type, idx in class_to_idx.items():
                classes[idx] = img_type
        
        return classes, class_to_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.

    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root
        ### список путей до картинок
        self.samples = []
        for img in os.listdir(self.root):
            self.samples.append(img)
        ### преобразования: ресайз + нормализация + ToTensorV2
        self.transform = A.Compose([
            A.Resize(height=232, width=232, interpolation=cv2.INTER_LINEAR),
            A.CenterCrop(height=MODEL_INPUT_SIZE[0], width=MODEL_INPUT_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1),
            ToTensorV2(),
        ])
        self.targets = None
        if annotations_file is not None:
            ### словарь, targets[путь до картинки] = индекс класса
            self.targets = {}
            classes, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
            filename_class_name = {}
            with open(annotations_file) as file:
                reader = csv.reader(file)
                next(reader)
                
                for row in reader:
                    filename_class_name[row[0]] = row[1]
                    
            for img in os.listdir(self.root):
                self.targets[img] = class_to_idx[filename_class_name[img]]

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        path_to_img = os.path.join(self.root, self.samples[index])
        class_idx = -1
        image = self.transform(image=np.asarray(Image.open(path_to_img).convert('RGB')))['image']
        if self.targets is not None:
            class_idx = self.targets[self.samples[index]]
            
        return image, self.samples[index], class_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)


class CustomNetwork(L.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.

    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """

    def __init__(
        self,
        features_criterion: (
            typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        internal_features: int = 1024,
        lr = 1e-3,
        transfer=False
    ):
        super().__init__()
        self.features_criterion = features_criterion
        if features_criterion is not None:
            self.features_criterion = features_criterion()
        self.internal_features = internal_features
        self.transfer = transfer
        self.model = self.get_model()
        self.loss_fn = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.lr = lr
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=CLASSES_CNT,
        )
        
    def get_model(self):
        weights = torchvision.models.ResNet50_Weights.DEFAULT if self.transfer else None
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Sequential(OrderedDict([
            (f'lin1', nn.Linear(in_features=model.fc.in_features, out_features=self.internal_features)),
            (f'batchnorm1', nn.BatchNorm1d(self.internal_features)),
            (f'act', nn.ReLU()),
            (f'lin2', nn.Linear(in_features=self.internal_features, out_features=CLASSES_CNT)),
        ]))
        
        if self.features_criterion is not None:
            model.fc = nn.Identity()
        
        for child in list(model.children())[:-4]:
            for param in child.parameters():
                param.requires_grad = False
        
        return model
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.01)

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "train_loss",
        }
        return [optimizer], [lr_scheduler]

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Функция для прогона данных через нейронную сеть.
        Возвращает два тензора: внутреннее представление и логиты после слоя-классификатора.
        """
        if self.features_criterion is not None:
            forward_res = self.model(x)
        else:
            forward_res = self.log_softmax(self.model(x))
        return forward_res
        

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param x: батч с картинками
        """
        return  self.forward(x).argmax(axis=-1)
    
    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")
    
    def _step(self, batch, kind):
        imgs, path, target = batch
        pred = self.forward(imgs)

        if self.features_criterion is not None:
            loss = self.features_criterion(pred, target)
            accs = None
        else:
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
            on_step=False,
            on_epoch=True,
        )
        if self.logger is not None:
            self.logger.experiment.add_scalars("losses", {f"{kind}_loss": loss}, self.global_step)
            if self.features_criterion is None:
                self.logger.experiment.add_scalars("accuracy", {f"{kind}_accs": accs}, self.global_step)
        return loss
    
    
def train_model(
    model,
    experiment_path,
    dl_train,
    dl_val,
    max_epochs=10,
    fast_train=False,
    ckpt_path=None,
    monitor="val_accs",
    **trainer_kwargs,
):
    callbacks = [
        # L.pytorch.callbacks.TQDMProgressBar(),
        L.pytorch.callbacks.LearningRateMonitor(),
        L.pytorch.callbacks.ModelCheckpoint(
            filename="{epoch}-{accs:.3f}",
            monitor=monitor,
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=4,
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
    trainer.fit(model, dl_train, dl_val, ckpt_path=ckpt_path)


def train_simple_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на исходных данных.
    """
    batch_size = 64
    num_workers = 2
    ds_train = DatasetRTSD(
        root_folders=['./cropped-train'], 
        path_to_classes_json='./classes.json'
    )
    dataloader_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    ds_val = TestData(
        root="./smalltest",
        path_to_classes_json='./classes.json',
        annotations_file="./smalltest_annotations.csv",
    )
    dataloader_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    
    simple_classifier = CustomNetwork(lr=1e-4, transfer=True)
    train_model(
        simple_classifier,
        "./simple_classifier",
        dataloader_train,
        dataloader_val,
        accelerator="auto",
        max_epochs=6,
    )
    
    torch.save(simple_classifier.state_dict(), './simple_model.pth')
    
    return simple_classifier


def apply_classifier(
    model: torch.nn.Module,
    test_folder: str,
    path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    """
    Функция, которая применяет модель и получает её предсказания.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    ### список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    results = []
    
    batch_size = 1
    num_workers = 0
    
    ds_test = TestData(
        test_folder,
        path_to_classes_json
    )
    dataloader_test = DataLoader(
        dataset=ds_test,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    classes, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            imgs, path, target = batch
            
            pred = model.predict(imgs)

            cur_res = {'filename': path[0], 'class': classes[pred[0]]}
            results.append(cur_res)
    # print(results)
    return results

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        reader = csv.DictReader(fhandle)
        for row in reader:
            res[row["filename"]] = row["class"]
    return res


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == "all" or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)

def test_classifier(
    model: torch.nn.Module,
    test_folder: str,
    annotations_file: str,
    classes_file = './classes.json',
) -> typing.Tuple[float, float, float]:
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    with open(classes_file, "r") as fr:
        classes_info = json.load(fr)
    class_name_to_type = {k: v["type"] for k, v in classes_info.items()}
    
    results = apply_classifier(model, test_folder, classes_file)
    results = {elem["filename"]: elem["class"] for elem in results}
    
    gt = read_csv(annotations_file)
    y_pred = []
    y_true = []
    for k, v in results.items():
        y_pred.append(v)
        y_true.append(gt[k])

    total_acc = calc_metric(y_true, y_pred, "all", class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, "rare", class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, "freq", class_name_to_type)
    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.

    :param background_path: путь до папки с изображениями фона
    """

    def __init__(self, background_path: str) -> None:
        super().__init__()
        self.bg_paths = []
        for bg in os.listdir(background_path):
            self.bg_paths.append(os.path.join(background_path, bg))

    ### Для каждого из необходимых преобразований над иконками/картинками,
    ### напишите вспомогательную функцию приблизительно следующего вида:
    ###
    ### @staticmethod
    ### def discombobulate_icon(icon: np.ndarray) -> np.ndarray:
    ###     ### YOUR CODE HERE
    ###     return ...
    ###
    ### Постарайтесь не использовать готовые библиотечные функции для
    ### аугментаций и преобразования картинок, а реализовать их
    ### "из первых принципов" на numpy
    
    @staticmethod
    def resize_icon(icon: np.ndarray) -> np.ndarray:
        size_h = np.random.randint(16, 129)
        size_w = np.random.randint(16, 129)
        return cv2.resize(icon, (size_h, size_w))
    
    @staticmethod
    def pad_icon(icon: np.ndarray) -> np.ndarray:
        pad = np.random.randint(0, 16) / 100
        pad_size_h = int(pad * icon.shape[0]) + 1
        pad_size_w = int(pad * icon.shape[1]) + 1
        return cv2.copyMakeBorder(icon, pad_size_h, pad_size_h, pad_size_w, pad_size_w, cv2.BORDER_CONSTANT, value=0)
    
    @staticmethod
    def change_color_icon(icon: np.ndarray) -> np.ndarray:
        image = Image.fromarray(icon).convert("RGBA")
        saturation_change = np.random.uniform(0.5, 1.3)
        image = PIL.ImageEnhance.Brightness(image).enhance(saturation_change)
        color = np.random.uniform(0.6, 0.9)
        image = PIL.ImageEnhance.Color(image).enhance(color)
        # hsv_img = cv2.cvtColor(icon[:, :, :-1], cv2.COLOR_RGB2HSV)
        # # Apply hue change
        # hue_change = np.random.uniform(-0.5, 0.5)
        # hsv_img[..., 0] = (hsv_img[..., 0] + hue_change) % 180
        # # Apply saturation
        # saturation_change = np.random.uniform(0.2, 0.3)
        # hsv_img[..., 1] = np.clip(hsv_img[..., 1] + saturation_change, 0, 255)
        # value_change = np.random.uniform(0.2, 0.3)
        # hsv_img[..., 2] = np.clip(hsv_img[..., 2] + value_change, 0, 255)
        
        # icon[:,:,:3] = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return np.asarray(image)
    
    # from https://stackforgeeks.com/blog/opencv-python-rotate-image-by-x-degrees-around-specific-point
    @staticmethod
    def rotate_icon(icon: np.ndarray, specific_angle = None) -> np.ndarray:
        angle = np.random.randint(-15, 16)
        if specific_angle is not None:
            angle = specific_angle
        image_center = tuple(np.array(icon.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(icon, rot_mat, icon.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    @staticmethod
    def blur_icon(icon: np.ndarray) -> np.ndarray:
        angle = np.random.randint(-90, 91)
        kernel_size = np.random.randint(3, 11)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        kernel = SignGenerator.rotate_icon(kernel, specific_angle=angle)
        
        icon[:, :, :3] = cv2.filter2D(icon[:, :, :3], -1, kernel)
        return icon
    
    @staticmethod
    def gauss_filter_icon(icon: np.ndarray) -> np.ndarray:
        rand_kernel = np.random.randint(3, 8)
        if rand_kernel % 2 != 1:
            rand_kernel += 1
        icon[:, :, :3] = cv2.GaussianBlur(icon[:, :, :3], ksize=(rand_kernel, rand_kernel), sigmaX=5.0)
        return icon
        
    
       

    def get_sample(self, icon: np.ndarray) -> np.ndarray:
        """
        Функция, встраивающая иконку на случайное изображение фона.

        :param icon: Массив с изображением иконки
        """
        # resize
        icon = SignGenerator.resize_icon(icon)
        # pad
        icon = SignGenerator.pad_icon(icon)
        # change color
        icon = SignGenerator.change_color_icon(icon)
        # rotate
        icon = SignGenerator.rotate_icon(icon)
        # blur
        icon = SignGenerator.blur_icon(icon)
        # gauss filter 
        icon = SignGenerator.gauss_filter_icon(icon)
        icon_size_h = icon.shape[0]
        icon_size_w = icon.shape[1]
        
        ### случайное изображение фона
        bg_idx = np.random.randint(0, len(self.bg_paths))
        bg_img = np.asarray(Image.open(self.bg_paths[bg_idx]).convert("RGB"))
        
        crop_transform = A.Compose([A.RandomCrop(height=icon_size_h, width=icon_size_w)])
        bg_img = crop_transform(image=bg_img)["image"]
        
        mask = icon[:, :, 3] / 255
        mask = np.stack([mask] * 3, axis=-1)
        img_with_bg = bg_img * (1 - mask) + icon[:, :, :3] * mask
        
        return img_with_bg.astype(np.uint8)


def generate_one_icon(args: typing.Tuple[str, str, str, int]) -> None:
    """
    Функция, генерирующая синтетические данные для одного класса.

    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    path_icon = args[0]
    dst_path = args[1]
    bg_path = args[2]
    cnt_imgs = args[3]
    
    class_name = os.path.split(path_icon)[-1][:-4]
    
    icon_generator = SignGenerator(bg_path)
    
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        
    if not os.path.exists(os.path.join(dst_path, class_name)):
        os.mkdir(os.path.join(dst_path, class_name))

    dst_path = os.path.join(dst_path, class_name)
    for idx in range(cnt_imgs):
        new_icon = icon_generator.get_sample(np.asarray(Image.open(path_icon).convert("RGBA")))
        Image.fromarray(new_icon).save(os.path.join(dst_path, f'{idx}.png')) 


def generate_all_data(
    output_folder: str,
    icons_path: str,
    background_path: str,
    samples_per_class: int = 1000,
) -> None:
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.

    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    shutil.rmtree(output_folder, ignore_errors=True)
    with ProcessPoolExecutor(8) as executor:
        params = [
            [
                os.path.join(icons_path, icon_file),
                output_folder,
                background_path,
                samples_per_class,
            ]
            for icon_file in os.listdir(icons_path)
        ]
        list(tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на смеси исходных и ситетических данных.
    """
    batch_size = 64
    num_workers = 2
    ds_train = DatasetRTSD(
        root_folders=['./cropped-train', './syntetic_icons'], 
        path_to_classes_json='./classes.json'
    )
    dataloader_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    ds_val = TestData(
        root="./smalltest",
        path_to_classes_json='./classes.json',
        annotations_file="./smalltest_annotations.csv",
    )
    dataloader_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    
    simple_classifier = CustomNetwork(lr=0.00001, transfer=True)
    train_model(
        simple_classifier,
        "./simple_with_synt_classifier",
        dataloader_train,
        dataloader_val,
        accelerator="auto",
        max_epochs=8,
    )
    
    torch.save(simple_classifier.state_dict(), './simple_model_with_synt.pth')
    
    return simple_classifier


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """

    def __init__(self) -> None:
        super().__init__()
        self.margin = 2.0
        self.eps = 1e-9

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Функция, вычисляющая loss-функцию на признаки предпоследнего слоя нейросети.

        :param outputs: Признаки с предпоследнего слоя нейросети
        :param labels: Реальные метки объектов
        """

        num_positive = 0
        num_negative = 0
        dist_pos = 0
        dist_neg = 0
        for i in range(0, len(outputs)):
            for j in range(0, len(outputs)):
                if labels[i] == labels[j]:
                    num_positive += 1
                    dist_pos += (outputs[i] - outputs[j]).square().sum()
                else:
                    num_negative += 1
                    dist_neg += F.relu(self.margin - ((outputs[i] - outputs[j]).square().sum() + self.eps).sqrt()).square()
                    
        if num_positive == 0:
            loss = 0.5 * dist_neg / num_negative
        elif num_negative == 0:
            loss = 0.5 * dist_pos / num_positive
        else:  
            loss = 0.5 * dist_pos / num_positive + 0.5 * dist_neg / num_negative
        return loss
        
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.

    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """

    def __init__(
        self,
        data_source: DatasetRTSD,
        elems_per_class: int,
        classes_per_batch: int,
    ) -> None:
        self.dataset = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        set_random_seed()

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        batch_size = self.classes_per_batch * self.elems_per_class
        N = len(self.dataset.samples) // batch_size
        
        all_batches = []
        for idx in range(N):
            unique_classes = random.sample(list(self.dataset.classes_to_samples.keys()), k=self.classes_per_batch)
            samples = []
            for class_id in unique_classes:
                cur_class_samples = self.dataset.classes_to_samples[class_id]
                if len(cur_class_samples) < self.elems_per_class:
                    samples_in_batch = cur_class_samples + random.choices(cur_class_samples, k=self.elems_per_class - len(cur_class_samples))
                else:
                    samples_in_batch = random.sample(cur_class_samples, k=self.elems_per_class)
                samples.append(samples_in_batch)
                
            samples = np.array(samples).flatten().tolist()
            all_batches.append(samples)
            
        return iter(all_batches)

    def __len__(self) -> None:
        """
        Возвращает общее количество батчей.
        """
        batch_size = self.classes_per_batch * self.elems_per_class
        return len(self.dataset.samples) // batch_size


def train_better_model() -> torch.nn.Module:
    """
    Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки.
    """
    batch_size = 64
    num_workers = 2
    ds_train = DatasetRTSD(
        root_folders=['./cropped-train', './syntetic_icons'], 
        path_to_classes_json='./classes.json'
    )
    sampler = CustomBatchSampler(
        data_source=ds_train,
        elems_per_class=5,
        classes_per_batch=30,
    )
    dataloader_train = DataLoader(
        dataset=ds_train,
        num_workers=num_workers,
        batch_sampler=sampler,
    )
    # ds_val = TestData(
    #     root="./smalltest",
    #     path_to_classes_json='./classes.json',
    #     annotations_file="./smalltest_annotations.csv",
    # )
    # dataloader_val = DataLoader(
    #     dataset=ds_val,
    #     num_workers=num_workers,
    #     batch_sampler=sampler,
    # )
    
    better_classifier = CustomNetwork(lr=1e-5, features_criterion=FeaturesLoss, transfer=True)
    train_model(
        better_classifier,
        "./improved_classifier",
        dataloader_train,
        None,
        accelerator="auto",
        max_epochs=3,
        monitor="train_loss"
    )
    
    torch.save(better_classifier.state_dict(), './improved_features_model.pth')
    
    return better_classifier


class ModelWithHead(CustomNetwork):
    """
    Класс, реализующий модель с головой из kNN.

    :param n_neighbors: Количество соседей в методе ближайших соседей
    """

    def __init__(self, n_neighbors: int, transfer = False) -> None:
        super().__init__()
        self.eval()
        self.n_neighbors = n_neighbors
        
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.emb_net = CustomNetwork(lr=1e-5, features_criterion=FeaturesLoss, transfer=transfer)

    def load_nn(self, nn_weights_path: str) -> None:
        """
        Функция, загружающая веса обученной нейросети.

        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        self.emb_net.load_state_dict(torch.load(nn_weights_path, weights_only=True))

    def load_head(self, knn_path: str) -> None:
        """
        Функция, загружающая веса kNN (с помощью pickle).

        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        with open(knn_path, "rb") as knn_model:
            self.knn = pickle.load(knn_model)

    def save_head(self, knn_path: str) -> None:
        """
        Функция, сохраняющая веса kNN (с помощью pickle).

        :param knn_path: Путь, куда надо сохранить веса kNN
        """
        with open(knn_path, 'wb') as knn_model:
            pickle.dump(self.knn, knn_model)

    def train_head(self, indexloader: torch.utils.data.DataLoader) -> None:
        """
        Функция, обучающая голову kNN.

        :param indexloader: Загрузчик данных для обучения kNN
        """
        self.emb_net.model.eval()
        with torch.no_grad():
            features = []
            targets = []
            
            for batch in tqdm(indexloader):
                imgs, path, target = batch
                
                features_last_layer = self.emb_net.forward(imgs).detach().cpu().numpy()
                features_last_layer = features_last_layer / np.linalg.norm(features_last_layer, axis=1)[:, None]
                features.append(features_last_layer)
                targets.append(target.detach().cpu().numpy())
                
        features = np.concatenate(features, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        self.knn.fit(features, targets)
        

    def predict(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param imgs: батч с картинками
        """
        
        self.emb_net.model.eval()
        with torch.no_grad():
            features_last_layer = self.emb_net.forward(imgs).detach().cpu().numpy()
            features_last_layer = features_last_layer / np.linalg.norm(features_last_layer, axis=1)[:, None]

        # self.knn.eval()
        with torch.no_grad():
            knn_pred = self.knn.predict(features_last_layer)
        # print(knn_pred.shape)
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.

    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """

    def __init__(self, data_source: DatasetRTSD, examples_per_class: int) -> None:
        self.dataset = data_source
        self.examples_per_class = examples_per_class

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        all_batches = []
        for key, value in self.dataset.classes_to_samples.items():
            cur_classes = random.sample(value, k=self.examples_per_class)
            all_batches.append(cur_classes)
            
        all_batches = np.array(all_batches).flatten().tolist()
        random.shuffle(all_batches)
        return iter(all_batches)
        

    def __len__(self) -> int:
        """
        Возвращает общее количество индексов.
        """
        return len(self.dataset.classes) * self.examples_per_class


def train_head(nn_weights_path: str, examples_per_class: int = 20) -> torch.nn.Module:
    """
    Функция для обучения kNN-головы классификатора.

    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    
    batch_size = 64
    num_workers = 2
    ds_train = DatasetRTSD(
        root_folders=['./syntetic_icons'], 
        path_to_classes_json='./classes.json'
    )
    sampler = IndexSampler(
        data_source=ds_train,
        examples_per_class=examples_per_class,
    )
    dataloader_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
    )
    
    model_with_head = ModelWithHead(n_neighbors=examples_per_class, transfer=False)
    model_with_head.load_nn(nn_weights_path)
    
    model_with_head.train_head(dataloader_train)
    
    model_with_head.save_head('./knn_model.bin')
    
    return model_with_head

if __name__ == "__main__":
    # The following code won't be run in the test system, but you can run it
    # on your local computer with `python -m rare_traffic_sign_solution`.

    # Feel free to put here any code that you used while
    # debugging, training and testing your solution.
    
    # I prefer .ipynb testing
    pass
