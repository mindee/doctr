# Formula recognition

The sample scripts in this folder let you train, evaluate and benchmark formula recognition models with docTR.
A formula recognition model reads the image of a mathematical formula and predicts its LaTeX source, token by token.

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

## Usage

You can start your training in PyTorch:

```shell
python references/formula/train.py formulanet --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
```

To try the pipeline end-to-end on a small toy dataset (32 formulas of UniMER-1M and UniMER-Test):

```shell
wget https://github.com/mindee/doctr/releases/download/v1.1.0/toy_formula_set-8f54029b.zip
unzip toy_formula_set-8f54029b.zip -d formula_set
python references/formula/train.py formulanet --train_path ./formula_set --val_path ./formula_set -b 2 --epochs 1
```

`--input_size H W` sets the input size of the model (`256 768` by default): each formula is resized to fit it while
keeping its aspect ratio, and padded. The training adds random margins around half of the formulas
(`doctr.transforms.RandomPad`: each side up to half the image height, in the background color of the image), so that the
model reads the crops of layout regions as they are: the formulas are not trimmed, neither at training nor at inference.

`--max_length` sets the maximum number of tokens of a formula (`4096` by default, enough for the longest UniMER-1M
label).

Long formulas are memory-hungry: a batch is padded to its longest formula. `--max-tokens` bounds the number of padded
target tokens per forward pass (`32768` by default, about 8.5 GB of GPU memory with `--amp-dtype bfloat16` and the
default batch size): a batch above it is split into chunks of formulas of similar length, whose gradients are
accumulated, so the update is the same as for the whole batch. Lower it if you run out of memory.

The validation decodes each formula greedily until its end token, or up to `--max_decode_length` tokens (`2048` by
default, enough for 99.99% of the UniMER-Test labels). The decoding of a batch lasts until its longest formula ends, so
early in a training, before the model learns to stop, a validation batch of 32 formulas can take about 3.5 s on an
RTX 4080.

### Multi-GPU support

We now use the built-in [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) launcher to spawn your DDP workers. `torchrun` will set all the necessary environment variables (`LOCAL_RANK`, `RANK`, etc.) for you. Arguments are the same than the ones from single GPU, except:

- `--backend`: you can specify another `backend` for `DistributedDataParallel` if the default one is not available on
your operating system. Fastest one is `nccl` according to [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

#### Key `torchrun` parameters

- `--nproc_per_node=<N>`
  Spawn `<N>` processes on the local machine (typically equal to the number of GPUs you want to use).
- `--nnodes=<M>`
  (Optional) Total number of nodes in your job. Default is 1.
- `--rdzv_backend`, `--rdzv_endpoint`, `--rdzv_id`
  (Optional) Rendezvous settings for multi-node jobs. See the [torchrun docs](https://pytorch.org/docs/stable/elastic/run.html) for details.

#### GPU selection

By default all visible GPUs will be used. To limit which GPUs participate, set the `CUDA_VISIBLE_DEVICES` environment variable **before** running `torchrun`. For example, to use only CUDA devices 0 and 2:

```shell
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --nproc_per_node=2 references/formula/train.py \
  formulanet \
  --train_path path/to/train \
  --val_path   path/to/val \
  --epochs 5 \
  --backend nccl
```

## Device and mixed precision

Every training and evaluation script accepts `--device`: a CUDA index (`0`), `cuda:N`, `mps` (Apple Silicon GPU) or `cpu`. Without it the script picks CUDA if available, then MPS, then CPU. In distributed mode (`torchrun`) the argument is ignored and each process uses its own GPU.

`--amp` enables automatic mixed precision and is only supported on CUDA. `--amp-dtype bfloat16` (Ampere or newer GPUs) uses bfloat16 instead of float16: it has the range of float32, so it needs no loss scaling and avoids the overflows float16 can produce in some losses.

```shell
# NVIDIA GPU with bfloat16 mixed precision
python references/formula/train.py formulanet --train_path path/to/train --val_path path/to/val --epochs 5 --device 0 --amp --amp-dtype bfloat16
```

## Checkpoints

Each run writes its metadata once, as `<experiment name>.json` next to the checkpoints it saves. It holds what is needed to rebuild the model for inference and to reproduce the run: the architecture and its task settings (`vocab`, `input_shape`, `max_length`, `max_decode_length`), the dataset hashes, the docTR / PyTorch versions, the git revision and the full list of arguments.

## Evaluation

You can evaluate a model (the pretrained one by default, or your own checkpoint with `--resume`) on a dataset:

```shell
python references/formula/evaluate.py formulanet path/to/your/dataset --resume path/to/your/checkpoint.pt
```

The script reports, with `doctr.utils.metrics.FormulaMetric`, on the LaTeX tokens (so the whitespaces of the labels do not matter):

- the **exact match** rate: the share of formulas whose tokens are all correct;
- the **edit similarity**: one minus the token-level edit distance, relative to the longest of the two sequences, averaged over the formulas;
- the corpus-level **BLEU-4** score.

To compare with published results, evaluate each subset of [UniMER-Test](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)
separately (`spe`, `cpe`, `sce`, `hwe`).

## Latency benchmark

You can measure the inference latency of an architecture:

```shell
python references/formula/latency.py formulanet --it 100 --size 256 768 --tokens 128 --gpu
```

The decoding time grows with the length of the formulas: `--tokens` sets the number of tokens decoded per formula.

## Data format

You need to provide both `train_path` and `val_path` arguments to start training (`evaluate.py` takes a single `dataset_path`).
Each path must lead to a folder with 1 subfolder and 1 file (`train_path` can also lead to a folder of such folders, which are merged):

```shell
├── images
│   ├── sample_img_01.png
│   ├── sample_img_02.png
│   ├── sample_img_03.png
│   └── ...
└── labels.json
```

`labels.json` maps each image file name to the LaTeX source of its formula, without math delimiters (`$`, `$$`, `\[`):

```shell
{
    "sample_img_01.png": "\\frac{a}{b^{2}}",
    "sample_img_02.png": "\\sum _ { i = 1 } ^ { n } x _ { i }",
    ...
}
```

The labels are split into tokens with `doctr.datasets.tokenize_latex`: commands (`\frac`, `\operatorname*`), environment
delimiters (`\begin{array}`), control symbols (`\{`, `\,`, `\\`) and single characters. They can be given raw or
already separated by spaces. The model predicts the tokens separated by spaces (e.g. `\frac { a } { b ^ { 2 } }`), which
renders the same in KaTeX or MathJax.

The samples holding a token out of the vocabulary, or more tokens than `--max_length`, are skipped (the scripts report
how many). For the best results, normalize the labels with the [KaTeX](https://katex.org/) parser before training
(e.g. matrices as `\begin{array}`), so that a formula has a single spelling.

### Vocabulary

The default vocabulary, `doctr.datasets.LATEX_TOKENS`, holds the printable ASCII characters, the commands and
environments supported by KaTeX, and every other token of the UniMER-1M, HME100K and UniMER-Test labels (TeX commands
such as `\ensuremath` or `\textsc`, Greek, Cyrillic and CJK characters...): it covers all of them, no `--vocab` is needed
to train on these datasets. To recognize other symbols, extend it with the tokens of your labels that occur often
enough, and train with it:

```shell
python references/formula/build_vocab.py path/to/train_set --output vocab.json --min-count 10
python references/formula/train.py formulanet --train_path path/to/train_set --val_path path/to/val_set --vocab vocab.json
```

Pass the same `--vocab` to `evaluate.py`, and the vocabulary to the model to use it (`formulanet(vocab=...)`).

## Slack Logging with tqdm

To enable Slack logging using `tqdm`, you need to set the following environment variables:

- `TQDM_SLACK_TOKEN`: the Slack Bot Token
- `TQDM_SLACK_CHANNEL`: you can retrieve it using `Right Click on Channel > Copy > Copy link`. You should get something like `https://xxxxxx.slack.com/archives/yyyyyyyy`. Keep only the `yyyyyyyy` part.

You can follow this page on [how to create a Slack App](https://api.slack.com/quickstart).

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```python
python references/formula/train.py --help
```
