# Adversarial Patch for 3D Local Feature Extractor

## Install packages
```
pip intsall -r requirements.txt
```

## Prepare data
- Download from data hpatches-sequences-release.tar.gz from [HPatches](icvl.ee.ic.ac.uk/vbalnt/hpatches/)


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

## Folder structure
```
.
├── chessboard.py
├── data
│   └── hpatches-sequences-release/
├── homography_transforms.py
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
└── requirements.txt
```