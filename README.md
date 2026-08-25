# Learning Shared-Specific Representations for Cross-Spectral Image Generation

Official implementation of **Learning Shared-Specific Representations for
Cross-Spectral Image Generation** (**ECCV 2026**).

SHASP translates an ordinary RGB visible-light (VIS) image into an infrared
(IR) image. Here, *cross-spectral* refers to the VIS and IR imaging modalities;
the input is not a hyperspectral cube. This repository contains the training,
inference, and evaluation code. Please refer to the paper for the specific
model structure and research methods. Datasets, checkpoints, logs, and generated
results are not included in Git history.

## Installation

Python 3.9+ and PyTorch 1.13+ are recommended. Install a PyTorch build matching
your CUDA version, then use either Conda or pip:

```bash
conda env create -f environment.yml
conda activate shasp
```

```bash
python -m pip install -r requirements.txt
```

Before a long training run, verify the installation with:

```bash
python scripts/smoke_test_shasp.py
```

## Dataset

Domain A is RGB VIS and domain B is grayscale IR:

```text
datasets/YourDataset/
├── trainA/   # RGB visible images
├── trainB/   # corresponding infrared images
├── testA/    # visible images
└── testB/    # optional infrared ground truth
```

In the default paired mode, A and B are matched by their positions after
lexicographic sorting. Both images in a pair receive exactly the same random
crop and horizontal flip. The loader requires equal training-set counts; use
matching, zero-padded names such as `000001.png` in both folders to make the
correspondence unambiguous. Source-only inference requires only `testA/`.
The datasets themselves are not redistributed by this repository; obtain them
from their official sources and follow their respective licenses and terms.

## Training

Run the recommended paired VIS-to-IR configuration with:

```bash
bash scripts/train_shasp.sh \
  ./datasets/YourDataset \
  shasp_yourdataset
```

The preset uses 512x512 crops, batch size 1, Adam with learning rate `1e-4`,
100 epochs at the initial rate, and 30 epochs of linear decay. Epoch headers
and losses are printed immediately and recorded in
`checkpoints/<experiment>/loss_log.txt`. Images shown in
`checkpoints/<experiment>/web/index.html` are `real_A`, `fake_B`, `real_B`,
`fake_A`, `rec_A`, and `rec_B`; the internal grayscale base is not published.

`fake_B` is the primary VIS-to-IR output. `fake_A` is the auxiliary IR-to-VIS
cycle output. During the default first five warm-up epochs, adversarial and
cycle losses are disabled and the reverse decoder starts from a zero residual,
so `fake_A` can appear almost identical to the grayscale `real_B`. This does
not mean that the A/B domains or HTML labels are swapped. Training samples are
shuffled, so images from different epochs are not necessarily the same scene.

The default adversarial objective is LSGAN. The logarithmic alternative is
available with `--gan_mode enhance`.

An existing compatible generator can initialize training through the optional
third argument:

```bash
bash scripts/train_shasp.sh \
  ./datasets/YourDataset \
  shasp_yourdataset \
  ./checkpoints/previous_experiment/latest_net_G.pth
```


## Inference

Inference requires a trained generator checkpoint at
`checkpoints/<experiment>/<epoch>_net_G.pth`; `latest_net_G.pth` is selected by
default.

```bash
bash scripts/test_shasp.sh \
  ./datasets/YourDataset \
  shasp_yourdataset
```

The default checkpoint is `latest_net_G.pth`. Extra test options may be
appended, for example `--epoch 100 --num_test 1000`. Results are written to

```text
results/shasp_yourdataset/test_latest/
├── index.html
└── images/
    ├── real_A/
    ├── fake_B/
    ├── real_B/   # when testB exists
    ├── fake_A/   # when testB exists
    ├── rec_A/    # when testB exists
    └── rec_B/    # when testB exists
```

Each output keeps the input stem and adds its visual label, for example
`000001_fake_B.png`. No `gray_A` directory or image is generated.

## Evaluation

For paired test data, compute SSIM and PSNR with:

```bash
python scripts/evaluate_shasp.py \
  --real_dir ./datasets/YourDataset/testB \
  --generated_dir ./results/shasp_yourdataset/test_latest/images/fake_B
```

Other indicators such as FID, KID, AP, etc. have not been placed in this repository.

## Poster
[6260_wang_1397×991mm.pdf](https://github.com/user-attachments/files/31427328/6260_wang_1397x991mm.pdf)

## Citation

```bibtex
@inproceedings{wang2026shasp,
  title     = {Learning Shared-Specific Representations for Cross-Spectral Image Generation},
  author    = {Wang, Haining and Li, Na and Zhao, Huijie and Da, Yifan and Wen, Yan and Su, Yi and Fang, Yuqiang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements and license

The training/test scaffolding and PatchGAN implementation are adapted from
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
This repository is released under the BSD 3-Clause License; third-party notices
are included in [LICENSE](LICENSE).
