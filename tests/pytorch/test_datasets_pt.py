import os
from shutil import move

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

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
        ['SVT', True, [512, 512], 100, True],
        ['SVT', False, [512, 512], 249, False],
        ['IC03', True, [512, 512], 246, True],
        ['IC03', False, [512, 512], 249, False],
    ],
)
def test_dataset(dataset_name, train, input_size, size, rotate):

    ds = datasets.__dict__[dataset_name](
        train=train, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
    )

    assert len(ds) == size
    assert repr(ds) == (f"{dataset_name}()" if train is None else f"{dataset_name}(train={train})")
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


def test_detection_dataset(mock_image_folder, mock_detection_label):

    input_size = (1024, 1024)

    ds = datasets.DetectionDataset(
        img_folder=mock_image_folder,
        label_path=mock_detection_label,
        sample_transforms=Resize(input_size),
    )

    assert len(ds) == 5
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32
    assert img.shape[-2:] == input_size
    # Bounding boxes
    assert isinstance(target, np.ndarray) and target.dtype == np.float32
    assert np.all(np.logical_and(target[:, :4] >= 0, target[:, :4] <= 1))
    assert target.shape[1] == 4

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
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
    assert isinstance(image, torch.Tensor)
    assert image.shape[-2:] == input_size
    assert image.dtype == torch.float32
    assert isinstance(label, str)

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, labels = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
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
    assert isinstance(img, torch.Tensor)
    assert img.shape[-2:] == input_size
    assert img.dtype == torch.float32
    # Bounding boxes
    assert isinstance(target['boxes'], np.ndarray) and target['boxes'].dtype == np.float32
    assert np.all(np.logical_and(target['boxes'][:, :4] >= 0, target['boxes'][:, :4] <= 1))
    assert target['boxes'].shape[1] == 5
    # Flags
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    # Cardinality consistency
    assert target['boxes'].shape[0] == len(target['labels'])

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
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
    assert isinstance(image, torch.Tensor)
    assert image.shape[-2:] == input_size
    assert image.dtype == torch.float32
    assert isinstance(label, int) and label < len(vocab)

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, torch.Tensor) and targets.shape == (2,)
    assert targets.dtype == torch.int64


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
    assert isinstance(img, torch.Tensor)
    assert img.shape[-2:] == input_size
    assert img.dtype == torch.float32
    assert isinstance(target, dict)
    assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    assert len(target['labels']) == len(target['boxes'])

    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
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
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, *input_size)
    assert img.dtype == torch.float32
    assert isinstance(target, dict)
    # depends on #702
    # assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    assert len(target['labels']) == len(target['boxes'])

    loader = DataLoader(
        ds, batch_size=2, drop_last=True, sampler=RandomSampler(ds), num_workers=0, pin_memory=True,
        collate_fn=ds.collate_fn)

    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


@pytest.mark.parametrize(
    "input_size, size, rotate",
    [
        [[512, 512], 3, True],  # Actual set has 626 training samples and 360 test samples
        [[512, 512], 3, False],
    ],
)
def test_sroie(input_size, size, rotate, mock_sroie_dataset):
    # monkeypatch the path to temporary dataset
    datasets.SROIE.TRAIN = (mock_sroie_dataset, None, "sroie2019_train_task1.zip")

    ds = datasets.SROIE(
        train=True, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
        cache_dir=mock_sroie_dataset, cache_subdir="sroie2019_train_task1",
    )

    assert len(ds) == size
    assert repr(ds) == f"SROIE(train={True})"
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, *input_size)
    assert img.dtype == torch.float32
    assert isinstance(target, dict)
    # depends on #702
    # assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    assert len(target['labels']) == len(target['boxes'])

    loader = DataLoader(
        ds, batch_size=2, drop_last=True, sampler=RandomSampler(ds), num_workers=0, pin_memory=True,
        collate_fn=ds.collate_fn)

    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


@pytest.mark.parametrize(
    "input_size, size, rotate",
    [
        [[512, 512], 3, True],  # Actual set has 149 training samples and 50 test samples
        [[512, 512], 3, False],
    ],
)
def test_funsd(input_size, size, rotate, mock_funsd_dataset):
    # monkeypatch the path to temporary dataset
    datasets.FUNSD.URL = mock_funsd_dataset
    datasets.FUNSD.SHA256 = None
    datasets.FUNSD.FILE_NAME = "funsd.zip"

    ds = datasets.FUNSD(
        train=True, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
        cache_dir=mock_funsd_dataset, cache_subdir="funsd",
    )

    assert len(ds) == size
    assert repr(ds) == f"FUNSD(train={True})"
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, *input_size)
    assert img.dtype == torch.float32
    assert isinstance(target, dict)
    # depends on #702
    # assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], tuple) and all(isinstance(s, str) for s in target['labels'])
    assert len(target['labels']) == len(target['boxes'])

    loader = DataLoader(
        ds, batch_size=2, drop_last=True, sampler=RandomSampler(ds), num_workers=0, pin_memory=True,
        collate_fn=ds.collate_fn)

    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


@pytest.mark.parametrize(
    "input_size, size, rotate",
    [
        [[512, 512], 3, True],  # Actual set has 800 training samples and 100 test samples
        [[512, 512], 3, False],
    ],
)
def test_cord(input_size, size, rotate, mock_cord_dataset):
    # monkeypatch the path to temporary dataset
    datasets.CORD.TRAIN = (mock_cord_dataset, None, "cord_train.zip")

    ds = datasets.CORD(
        train=True, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
        cache_dir=mock_cord_dataset, cache_subdir="cord_train",
    )

    assert len(ds) == size
    assert repr(ds) == f"CORD(train={True})"
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, *input_size)
    assert img.dtype == torch.float32
    assert isinstance(target, dict)
    # depends on #702
    # assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], tuple) and all(isinstance(s, str) for s in target['labels'])
    assert len(target['labels']) == len(target['boxes'])

    loader = DataLoader(
        ds, batch_size=2, drop_last=True, sampler=RandomSampler(ds), num_workers=0, pin_memory=True,
        collate_fn=ds.collate_fn)

    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


@pytest.mark.parametrize(
    "input_size, size, rotate",
    [
        [[512, 512], 2, True],  # Actual set has 772875 training samples and 85875 test samples
        [[512, 512], 2, False],
    ],
)
def test_synthtext(input_size, size, rotate, mock_synthtext_dataset):
    # monkeypatch the path to temporary dataset
    datasets.SynthText.URL = mock_synthtext_dataset
    datasets.SynthText.SHA256 = None
    datasets.SynthText.FILE_NAME = "SynthText.zip"

    ds = datasets.SynthText(
        train=True, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
        cache_dir=mock_synthtext_dataset, cache_subdir="SynthText",
    )

    assert len(ds) == size
    assert repr(ds) == f"SynthText(train={True})"
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, *input_size)
    assert img.dtype == torch.float32
    assert isinstance(target, dict)
    assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    assert len(target['labels']) == len(target['boxes'])

    loader = DataLoader(
        ds, batch_size=2, drop_last=True, sampler=RandomSampler(ds), num_workers=0, pin_memory=True,
        collate_fn=ds.collate_fn)

    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


@pytest.mark.parametrize(
    "input_size, size, rotate",
    [
        [[512, 512], 3, True],  # Actual set has 2700 training samples and 300 test samples
        [[512, 512], 3, False],
    ],
)
def test_artefact_detection(input_size, size, rotate, mock_doc_artefacts):
    # monkeypatch the path to temporary dataset
    datasets.DocArtefacts.URL = mock_doc_artefacts
    datasets.DocArtefacts.SHA256 = None
    datasets.DocArtefacts.FILE_NAME = "artefact_detection.zip"

    ds = datasets.DocArtefacts(
        train=True, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
        cache_dir=mock_doc_artefacts, cache_subdir="artefact_detection",
    )

    assert len(ds) == size
    assert repr(ds) == f"DocArtefacts(train={True})"
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, *input_size)
    assert img.dtype == torch.float32
    assert isinstance(target, dict)
    assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], np.ndarray)
    assert len(target['labels']) == len(target['boxes'])

    loader = DataLoader(
        ds, batch_size=2, drop_last=True, sampler=RandomSampler(ds), num_workers=0, pin_memory=True,
        collate_fn=ds.collate_fn)

    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (2, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


@pytest.mark.parametrize(
    "input_size, size, rotate",
    [
        [[32, 128], 1, True],  # Actual set has 2000 training samples and 3000 test samples
        [[32, 128], 1, False],
    ],
)
def test_iiit5k(input_size, size, rotate, mock_iiit5k_dataset):
    # monkeypatch the path to temporary dataset
    datasets.IIIT5K.URL = mock_iiit5k_dataset
    datasets.IIIT5K.SHA256 = None
    datasets.IIIT5K.FILE_NAME = "IIIT5K-Word-V3.tar"

    ds = datasets.IIIT5K(
        train=True, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate,
        cache_dir=mock_iiit5k_dataset, cache_subdir="IIIT5K",
    )

    assert len(ds) == size
    assert repr(ds) == f"IIIT5K(train={True})"
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, *input_size)
    assert img.dtype == torch.float32
    assert isinstance(target, dict)
    # depends on #702
    # assert isinstance(target['boxes'], np.ndarray) and np.all((target['boxes'] <= 1) & (target['boxes'] >= 0))
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    assert len(target['labels']) == len(target['boxes'])

    loader = DataLoader(
        ds, batch_size=1, drop_last=True, sampler=RandomSampler(ds), num_workers=0, pin_memory=True,
        collate_fn=ds.collate_fn)

    images, targets = next(iter(loader))
    assert isinstance(images, torch.Tensor) and images.shape == (1, 3, *input_size)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)
