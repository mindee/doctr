import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from copy import deepcopy

from doctr import datasets
from doctr.transforms import Resize


def test_visiondataset():
    url = 'https://data.deepai.org/mnist.zip'
    with pytest.raises(ValueError):
        datasets.datasets.VisionDataset(url, download=False)

    dataset = datasets.datasets.VisionDataset(url, download=True, extract_archive=True)
    assert len(dataset) == 0
    assert repr(dataset) == 'VisionDataset()'


@pytest.mark.parametrize(
    "dataset_name, train, input_size, size, rotate",
    [
        ['FUNSD', True, [512, 512], 149, False],
        ['FUNSD', False, [512, 512], 50, True],
        ['SROIE', True, [512, 512], 626, False],
        ['SROIE', False, [512, 512], 360, False],
        ['CORD', True, [512, 512], 800, True],
        ['CORD', False, [512, 512], 100, False],
    ],
)
def test_dataset(dataset_name, train, input_size, size, rotate):

    ds = datasets.__dict__[dataset_name](
        train=train, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate
    )

    assert len(ds) == size
    assert repr(ds) == f"{dataset_name}(train={train})"
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, *input_size)
    assert img.dtype == torch.float32
    assert isinstance(target, dict)

    loader = DataLoader(
        ds, batch_size=2, drop_last=True, sampler=RandomSampler(ds), num_workers=0, pin_memory=True,
        collate_fn=ds.collate_fn)

    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)

    # FP16 checks
    ds = datasets.__dict__[dataset_name](train=train, download=True, fp16=True)
    img, target = ds[0]
    assert img.dtype == torch.float16


def test_detection_dataset(mock_image_folder, mock_detection_label):

    input_size = (1024, 1024)

    ds = datasets.DetectionDataset(
        img_folder=mock_image_folder,
        label_folder=mock_detection_label,
        sample_transforms=Resize(input_size),
    )

    assert len(ds) == 5
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32
    assert img.shape[-2:] == input_size
    # Bounding boxes
    assert isinstance(target['boxes'], np.ndarray) and target['boxes'].dtype == np.float32
    assert np.all(np.logical_and(target['boxes'][:, :4] >= 0, target['boxes'][:, :4] <= 1))
    assert target['boxes'].shape[1] == 4
    # Flags
    assert isinstance(target['flags'], np.ndarray) and target['flags'].dtype == np.bool
    # Cardinality consistency
    assert target['boxes'].shape[0] == target['flags'].shape[0]

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)

    # Rotated DS
    rotated_ds = datasets.DetectionDataset(
        img_folder=mock_image_folder,
        label_folder=mock_detection_label,
        sample_transforms=Resize(input_size),
        rotated_bbox=True
    )
    _, r_target = rotated_ds[0]
    assert r_target['boxes'].shape[1] == 5

    # FP16
    ds = datasets.DetectionDataset(img_folder=mock_image_folder, label_folder=mock_detection_label, fp16=True)
    img, target = ds[0]
    assert img.dtype == torch.float16
    # Bounding boxes
    assert target['boxes'].dtype == np.float16


def test_recognition_dataset(mock_image_folder, mock_recognition_label):
    input_size = (32, 128)
    ds = datasets.RecognitionDataset(
        img_folder=mock_image_folder,
        labels_path=mock_recognition_label,
        sample_transforms=Resize(input_size, preserve_aspect_ratio=True),
    )
    assert len(ds) == 5
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[-2:] == input_size
    assert image.dtype == torch.float32
    assert isinstance(label, str)

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, labels = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(labels, list) and all(isinstance(elt, str) for elt in labels)

    # FP16
    ds = datasets.RecognitionDataset(img_folder=mock_image_folder, labels_path=mock_recognition_label, fp16=True)
    image, label = ds[0]
    assert image.dtype == torch.float16
    ds2, ds3 = deepcopy(ds), deepcopy(ds)
    ds2.merge_dataset(ds3)
    assert len(ds2) == 2 * len(ds)


def test_ocrdataset(mock_ocrdataset):

    input_size = (512, 512)

    ds = datasets.OCRDataset(
        *mock_ocrdataset,
        sample_transforms=Resize(input_size),
    )
    rotated_ds = datasets.OCRDataset(
        *mock_ocrdataset,
        sample_transforms=Resize(input_size),
        rotated_bbox=True
    )
    assert len(ds) == 5
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[-2:] == input_size
    assert img.dtype == torch.float32
    # Bounding boxes
    assert isinstance(target['boxes'], np.ndarray) and target['boxes'].dtype == np.float32
    assert np.all(np.logical_and(target['boxes'][:, :4] >= 0, target['boxes'][:, :4] <= 1))
    assert target['boxes'].shape[1] == 4
    _, r_target = rotated_ds[0]
    assert r_target['boxes'].shape[1] == 5
    # Flags
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    # Cardinality consistency
    assert target['boxes'].shape[0] == len(target['labels'])

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)

    # FP16
    ds = datasets.OCRDataset(*mock_ocrdataset, fp16=True)
    img, target = ds[0]
    assert img.dtype == torch.float16
    # Bounding boxes
    assert target['boxes'].dtype == np.float16
