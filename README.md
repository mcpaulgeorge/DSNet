# DSNet

This repository contains an implementation of **DSNet**, a dual-domain spiking network for image demoireing in the **RAW + sRGB** domains. The project is built on top of the BasicSR training/testing pipeline and focuses on efficient moire artifact removal with spiking neural networks.


The core idea is **moire-driven sparsity**: moire-corrupted regions tend to trigger denser spike activity than clean regions, so the network can spend more computation on artifact-heavy areas. According to the paper, DSNet achieves competitive restoration quality while reducing estimated energy consumption by 16%.

Related figures from the paper:

- [Motivation figure](figure/motivation.tiff)
- [Network overview](figure/network.tiff)

## Method Overview

DSNet jointly processes moire-corrupted **RAW** and **sRGB** inputs with two parallel branches and then fuses them for final restoration. The main components are:

- **CLIF blocks** for sparse spiking feature extraction
- **Dual-domain encoder-decoder branches** for RAW and sRGB representations
- **Cross-attention fusion** to align complementary information across domains
- **Color Refinement Modules (CRM)** to correct moire-induced color distortion

The main architecture is defined in `vd/archs/rrid_arch.py`.

## Repository Layout

- `train.py`: training entry point
- `test.py`: evaluation / inference entry point
- `options/train/Train.yml`: default training configuration
- `options/test/Test.yml`: default testing configuration
- `vd/archs/rrid_arch.py`: DSNet network definition
- `vd/data/rawrgb_vd_paired_dataset.py`: paired RAW+sRGB dataset loader
- `vd/models/rawrgbid_model.py`: model, validation, and visualization logic

## Environment

The current codebase expects a CUDA-enabled PyTorch environment. The key dependencies visible in this repository are:

- `basicsr==1.4.2`
- `scikit-image==0.15.0`
- `deepspeed`
- `spikingjelly`
- `lpips`
- `cupy`

Notes:

- The model explicitly uses the `cupy` backend in SpikingJelly.
- GPU execution is the expected setup for both training and testing.
- No pinned `requirements.txt` is included in this folder, so you may need to align package versions with your local CUDA / PyTorch environment.

## Data Preparation

The provided YAML files expect the dataset under:

```text
dataset/raw_moire_image_dataset/
├── trainset/
│   ├── gt_RGB
│   ├── gt_RAW_npz
│   ├── moire_RGB
│   └── moire_RAW_npz
└── testset/
    ├── gt_RGB
    ├── gt_RAW_npz
    ├── moire_RGB
    └── moire_RAW_npz
```


Examples:

- `0001_gt.png`: clean RGB target
- `0001_gt.npz`: clean RAW target
- `0001_m.png`: moire-corrupted RGB input
- `0001_m.npz`: moire-corrupted RAW input

RAW files are loaded from `*.npz` and should contain a `patch_data` array. See `vd/data/data_util.py` for the exact loading logic.

## How To Test

Before testing, update the checkpoint path in `options/test/Test.yml`:

```yaml
path:
  pretrain_network_g: ./experiments/DSNet/models/net_g_160000.pth
```

Then run:

```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/Test.yml
```

By default, test logs and visualizations are written under `results/DSNet/`.

## How To Train

First, check that the dataset paths in `options/train/Train.yml` match your local layout.

Single-GPU training:

```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/Train.yml
```

Distributed training:

```bash
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/Train.yml --launcher pytorch
```

By default, training logs, checkpoints, copied option files, and intermediate visualizations are written under `experiments/DSNet/`.

## Notes

- `options/test/Test.yml` and `options/train/Train.yml` both uses `network_g.type: DSNet`, which matches the registered architecture in `vd/archs/rrid_arch.py`.

