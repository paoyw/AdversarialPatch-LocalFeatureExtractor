from argparse import ArgumentParser
import itertools
import json
import os
from typing import Any

import numpy as np
import cv2
import torch
from scipy.spatial import ConvexHull, Delaunay

from models.superpoint import SuperPointNet


def in_convex_hull(point, hull: ConvexHull) -> bool:
    """
    Checks if the point is in the convex hull.

    Args:
        point: The point to be checked.
        hull: The convex hull to be checked.

    Returns:
        Returns if the point is in the convex hull
    """
    hull = Delaunay(hull.points)
    return hull.find_simplex(point) != -1


def interpolation(point, h, w):
    """Computes the nearest four neighbors with the coresponding weights for bilinear interpolation."""
    points = []
    weights = []
    i_min = np.clip(np.floor(point[1]), 0, h - 1)
    i_max = np.clip(np.ceil(point[1]), 0, h - 1)
    j_min = np.clip(np.floor(point[0]), 0, w - 1)
    j_max = np.clip(np.ceil(point[0]), 0, w - 1)
    for i, j in itertools.product([i_min, i_max], [j_min, j_max]):
        points.append([int(i), int(j)])
        weights.append((1 - np.abs(j - point[0])) * (1 - np.abs(i - point[1])))
    return points, weights


def fill_patch(img: torch.Tensor,
               patch: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    Fills the image with the patch according the position of the mask.

    Args:
        img: The image to be filled. The shape is [C, H, W].
        patch: The patch to be used. The shape is [C, H, W].
        mask: The position to be filled in the image. The shape is [4, 2].

    Returns:
        The result image.
    """
    assert (img.dim() == 3)
    assert (patch.dim() == 3)
    assert (mask.dim() == 2 and mask.shape[0] == 4 and mask.shape[1] == 2)

    img = torch.clone(img)

    _, h, w = img.shape
    _, patch_h, patch_w = patch.shape
    src_pts = mask.numpy()
    dst_pts = np.array([[0, 0],
                        [patch_w - 1, 0],
                        [patch_w - 1, patch_h - 1],
                        [0, patch_h - 1]])
    H, _ = cv2.findHomography(src_pts, dst_pts)

    mask_hull = ConvexHull(src_pts)

    i_start = int(mask[:, 1].min().clamp(0, h - 1))
    i_end = int(mask[:, 1].max().clamp(0, h - 1))
    j_start = int(mask[:, 0].min().clamp(0, w - 1))
    j_end = int(mask[:, 0].max().clamp(0, w - 1))

    for i in range(i_start, i_end + 1):
        for j in range(j_start, j_end + 1):
            src_pt = np.array([j, i, 1], dtype=np.float32)
            if not in_convex_hull(src_pt[:2], mask_hull):
                continue
            dst_pt = (H @ src_pt.T).T.squeeze()
            dst_pt = dst_pt[:2] / dst_pt[-1:]
            pts, weights = interpolation(dst_pt, patch_h, patch_w)
            img[:, i, j] = 0
            for pt, weight in zip(pts, weights):
                img[:, i, j] += patch[:, pt[1], pt[0]] * weight
    return img


def instance_eval(source_view: torch.Tensor,
                  source_mask: torch.Tensor,
                  target_view: torch.Tensor,
                  target_mask: torch.Tensor,
                  model: Any, device: str = 'cpu',
                  matrics: dict = {}) -> dict:
    """
    Evaluates an instance of an adversarial patch attack for the local feature extractor.

    Args:
        source_view: The image from the source view.
        source_mask: The source mask in the `source_view` image.
        target_view: The image from the target view.
        target_mask: The target mask in the `target_view` image.
        model: The targeted local feature extractor.
        device: The computational device.

    Returns:
        A dictionary of different metrics.
    """


def dir_eval(dir: str, mask_file: str, model: Any, device: str = 'cpu'):
    """
    Evaluates every instance with `instance_eval` in a directory.

    Args:
        dir: The directory to be evaluated.
        mask_file: The masking file.
        model: The targeted local feature extractor.
        device: The computational device.

    Returns:
        A dictionary of different metrics.
    """
    with open(os.path.join(dir, mask_file)) as f:
        mask_datas = json.load(f)


def main(args):
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
