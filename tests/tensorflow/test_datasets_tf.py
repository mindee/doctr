import os
from shutil import move

import numpy as np
import pytest
import tensorflow as tf

from doctr import datasets
from doctr.datasets import DataLoader
from doctr.transforms import Resize


@pytest.mark.parametrize(
    "dataset_name, train, input_size, size, rotate",
    [
        ['FUNSD', True, [512, 512], 149, False],
        ['FUNSD', False, [512, 512], 50, True],
        ['SROIE', True, [512, 512], 626, True],
        ['SROIE', False, [512, 512], 360, False],
        ['CORD', True, [512, 512], 800, True],
        ['CORD', False, [512, 512], 100, False],
        ['DocArtefacts', True, [512, 512], 2700, False],
        ['DocArtefacts', False, [512, 512], 300, True],
        ['IIIT5K', True, [32, 128], 2000, True],
        ['IIIT5K', False, [32, 128], 3000, False],
        ['SVT', True, [512, 512], 100, True],
        ['SVT', False, [512, 512], 249, False],
        ['SynthText', True, [512, 512], 27, True],  # Actual set has 772875 samples
        ['SynthText', False, [512, 512], 3, False],  # Actual set has 85875 samples
        ['IC03', True, [512, 512], 246, True],
        ['IC03', False, [512, 512], 249, False],
    ],
)
def test_dataset(dataset_name, train, input_size, size, rotate):

    if dataset_name.lower() == "synthtext":
        # Monkeypatch the class to download a subsample
        datasets.__dict__[
            dataset_name
        ].URL = 'https://github.com/mindee/doctr/releases/download/v0.4.1/synthtext_samples-89fd1445.zip'
        datasets.__dict__[dataset_name].SHA256 = '89fd1445457b9ad8391e17620c6ae1b45134be2bf5449f36e7e4275176cc16ac'

    ds = datasets.__dict__[dataset_name](
        train=train, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
    )

    assert len(ds) == size
    assert repr(ds) == (f"{dataset_name}()" if train is None else f"{dataset_name}(train={train})")
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape == (*input_size, 3)
    assert img.dtype == tf.float32
    assert isinstance(target, dict)

    loader = datasets.DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


def test_detection_dataset(mock_image_folder, mock_detection_label):

    input_size = (1024, 1024)

    ds = datasets.DetectionDataset(
        img_folder=mock_image_folder,
        label_path=mock_detection_label,
        sample_transforms=Resize(input_size),
    )

    assert len(ds) == 5
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape[:2] == input_size
    assert img.dtype == tf.float32
    # Bounding boxes
    assert isinstance(target, np.ndarray) and target.dtype == np.float32
    assert np.all(np.logical_and(target[:, :4] >= 0, target[:, :4] <= 1))
    assert target.shape[1] == 4

    loader = DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, np.ndarray) for elt in targets)

    # Rotated DS
    rotated_ds = datasets.DetectionDataset(
        img_folder=mock_image_folder,
        label_path=mock_detection_label,
        sample_transforms=Resize(input_size),
        rotated_bbox=True
    )
    _, r_target = rotated_ds[0]
    assert r_target.shape[1] == 5

    # File existence check
    img_name, _ = ds.data[0]
    move(os.path.join(ds.root, img_name), os.path.join(ds.root, "tmp_file"))
    with pytest.raises(FileNotFoundError):
        datasets.DetectionDataset(mock_image_folder, mock_detection_label)
    move(os.path.join(ds.root, "tmp_file"), os.path.join(ds.root, img_name))


def test_recognition_dataset(mock_image_folder, mock_recognition_label):
    input_size = (32, 128)
    ds = datasets.RecognitionDataset(
        img_folder=mock_image_folder,
        labels_path=mock_recognition_label,
        sample_transforms=Resize(input_size, preserve_aspect_ratio=True),
    )
    assert len(ds) == 5
    image, label = ds[0]
    assert isinstance(image, tf.Tensor)
    assert image.shape[:2] == input_size
    assert image.dtype == tf.float32
    assert isinstance(label, str)

    loader = DataLoader(ds, batch_size=2)
    images, labels = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(labels, list) and all(isinstance(elt, str) for elt in labels)

    # File existence check
    img_name, _ = ds.data[0]
    move(os.path.join(ds.root, img_name), os.path.join(ds.root, "tmp_file"))
    with pytest.raises(FileNotFoundError):
        datasets.RecognitionDataset(mock_image_folder, mock_recognition_label)
    move(os.path.join(ds.root, "tmp_file"), os.path.join(ds.root, img_name))


def test_ocrdataset(mock_ocrdataset):

    input_size = (512, 512)

    ds = datasets.OCRDataset(
        *mock_ocrdataset,
        sample_transforms=Resize(input_size),
    )
    assert len(ds) == 3
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.dtype == tf.float32
    assert img.shape[:2] == input_size
    # Bounding boxes
    assert isinstance(target['boxes'], np.ndarray) and target['boxes'].dtype == np.float32
    assert np.all(np.logical_and(target['boxes'][:, :4] >= 0, target['boxes'][:, :4] <= 1))
    assert target['boxes'].shape[1] == 5
    # Flags
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    # Cardinality consistency
    assert target['boxes'].shape[0] == len(target['labels'])

    loader = DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)

    # File existence check
    img_name, _ = ds.data[0]
    move(os.path.join(ds.root, img_name), os.path.join(ds.root, "tmp_file"))
    with pytest.raises(FileNotFoundError):
        datasets.OCRDataset(*mock_ocrdataset)
    move(os.path.join(ds.root, "tmp_file"), os.path.join(ds.root, img_name))


def test_charactergenerator():

    input_size = (32, 32)
    vocab = 'abcdef'

    ds = datasets.CharacterGenerator(
        vocab=vocab,
        num_samples=10,
        cache_samples=True,
        sample_transforms=Resize(input_size),
    )

    assert len(ds) == 10
    image, label = ds[0]
    assert isinstance(image, tf.Tensor)
    assert image.shape[:2] == input_size
    assert image.dtype == tf.float32
    assert isinstance(label, int) and label < len(vocab)

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, tf.Tensor) and targets.shape == (2,)
    assert targets.dtype == tf.int32


@pytest.mark.parametrize(
    "size, rotate",
    [
        [5, True],  # Actual set has 229 train and 233 test samples
        [5, False]

    ],
)
def test_ic13_dataset(mock_ic13, size, rotate):
    input_size = (512, 512)
    ds = datasets.IC13(
        *mock_ic13,
        sample_transforms=Resize(input_size),
        rotated_bbox=rotate,
    )

    assert len(ds) == size
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape[:2] == input_size
    assert img.dtype == tf.float32
    assert isinstance(target, dict)
    assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])

    loader = DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


@pytest.mark.parametrize(
    "input_size, size, rotate",
    [
        [[32, 128], 3, True],  # Actual set has 33402 training samples and 13068 test samples
        [[32, 128], 3, False],
    ],
)
def test_svhn(input_size, size, rotate, mock_svhn_dataset):
    # monkeypatch the path to temporary dataset
    datasets.SVHN.TRAIN = (mock_svhn_dataset, None, "svhn_train.tar")

    ds = datasets.SVHN(
        train=True, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
        cache_dir=mock_svhn_dataset, cache_subdir="svhn",
    )

    assert len(ds) == size
    assert repr(ds) == f"SVHN(train={True})"
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape == (*input_size, 3)
    assert img.dtype == tf.float32
    assert isinstance(target, dict)

    loader = datasets.DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)
