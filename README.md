# Adversarial Patch for 3D Local Feature Extractor

## Install packages
```
pip intsall -r requirements.txt
```

## Prepare data
- Download from data hpatches-sequences-release.tar.gz from [HPatches](icvl.ee.ic.ac.uk/vbalnt/hpatches/)


## Folder structure
```
.
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