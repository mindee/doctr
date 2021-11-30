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
        ['FUNSD', True, [512, 512], 149, False],
        ['FUNSD', False, [512, 512], 50, True],
        ['SROIE', True, [512, 512], 626, False],
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
