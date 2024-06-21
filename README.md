# Adversarial Patch for 3D Local Feature Extractor

[![arXiv](https://img.shields.io/badge/arXiv-2406.08102-b31b1b.svg)](https://arxiv.org/abs/2406.08102)

[![slide](https://img.shields.io/badge/google%20slide-presentation-ffba00.svg)](https://docs.google.com/presentation/d/e/2PACX-1vS7ybtX8od70jtzw3ggknzAB1CgHYkvqZMPKGhsuv28gr09hEmzewgKnIQqAbeo0bbzmNFzq7isCqb4/pub?start=false&loop=false&delayms=3000)

## Install packages
```
pip intsall -r requirements.txt
```

## Prepare data
- Download the data with the mask annotations from [data.tar.gz](https://drive.google.com/file/d/1vR04XRnptLyJtYn8KbPd7KhKadKeSQzX/view?usp=drive_link)

- Or download the original HPatches data from data hpatches-sequences-release.tar.gz from [HPatches](icvl.ee.ic.ac.uk/vbalnt/hpatches/), and generate your own mask by `maskgen.py`.

## Patch generation
### Chessboard pattern
```bash
python3 chessboard.py [-h] [--rect_width RECT_WIDTH] [--rect_height RECT_HEIGHT]
                      [--width WIDTH] [--height HEIGHT] [--save SAVE]
```

### Adversarial patch
```bash
python3 patchgen.py [-h] [--cuda] [--epoch EPOCH] [--width WIDTH]
                    [--height HEIGHT] [--decay DECAY] [--alpha ALPHA]
                    [--multiplier MULTIPLIER] [--save SAVE] [--model MODEL]
                    [--untargeted] [--init INIT] [--init-pattern INIT_PATTERN]
                    [--prob PROB]
```

## Mask generation
```bash
python3 maskgen.py [-h] [--dirs [DIRS ...]] [--dir DIR] [--mask-file MASK_FILE]
                   [--patch-width PATCH_WIDTH] [--patch-height PATCH_HEIGHT]
                   [--H H] [--individual] [--overlapping]
```

## Evaluate the attack
```bash
python3 patch_eval.py [-h] [--dirs [DIRS ...]] [--dir DIR] [--mask-file MASK_FILE]
                      [--null-mask] [--match-point-num MATCH_POINT_NUM]
                      [--patch-file PATCH_FILE] [--model MODEL] [--model-weight MODEL_WEIGHT]
                      [--device DEVICE] [--log LOG]
```
eg.
- Evaluates the chess-init with size 128 while attacking the targeted viewpoints using masking size 128.
```bash
python3 patch_eval.py \
    --dirs data/hpatches-mask/* \
    --mask-file mask_indiv_128.json \
    --patch-file patches/patch_chess_w128_h128.png \
    --device cuda \
    --log log/
```
- Evaluates the chessboard pattern with size 100 while attacking the untargeted viewpoints using masking size 128 and the model is SIFT.
```bash
python3 patch_eval.py \
    --dirs data/hpatches-mask/* \
    --mask-file mask_128.json \
    --patch-file patches/chessboard_128.png \
    --model sift \
    --device cuda \
    --log log/
```

## Folder structure
```
.
├── chessboard.py
├── data
│   └── hpatches-mask
│       ├── v_abstract/
│       └── ...
├── homography_transforms.py
├── LICENSE
├── maskgen.py
├── models
│   ├── __init__.py
│   ├── sift.py
│   ├── superpoint.py
│   └── superpoint_v1.pth
├── patches
│   ├── chessboard_100.png
│   └── ...
├── patch_eval.py
├── patchgen.py
├── README.md
├── report.pdf
└── requirements.txt
```
