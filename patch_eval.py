from argparse import ArgumentParser
import itertools
import json
import os
from typing import Any

import numpy as np
import cv2
import torch
from torchvision.transforms import v2
from scipy.spatial import ConvexHull, Delaunay

import homography_transforms as htfm
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
                  model: Any, device: str = 'cpu') -> dict:
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
        mismatch_rate: The matching ratio from the source mask in the source
        image to the target mask in the target image.
    """
    source_view = source_view.to(device).unsqueeze(dim=0)
    target_view = target_view.to(device).unsqueeze(dim=0)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        source_view_semi, source_view_desc = model(
            v2.functional.rgb_to_grayscale(source_view))
        target_view_semi, target_view_desc = model(
            v2.functional.rgb_to_grayscale(target_view))
    source_view_pt, source_view_desc = model.to_cv2(
        source_view_semi.squeeze(), source_view_desc.squeeze()
    )
    target_view_pt, target_view_desc = model.to_cv2(
        target_view_semi.squeeze(), target_view_desc.squeeze()
    )
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(source_view_desc, target_view_desc, k=2)
    matches = sorted(matches,
                     key=lambda m: np.abs(m[0].distance - m[1].distance))
    matches = matches[:1000]
    count = 0
    total = 0
    for m, n in matches:
        if in_convex_hull(source_view_pt[m.queryIdx].pt, ConvexHull(source_mask)):
            total += 1
            if in_convex_hull(target_view_pt[m.trainIdx].pt, ConvexHull(target_mask)):
                count += 1
    mismatch_rate = count / total
    return mismatch_rate


def dir_eval(dir: str, mask_file: str, patch_file: str,
             model: Any, device: str = 'cpu'):
    """
    Evaluates every instance with `instance_eval` in a directory.

    Args:
        dir: The directory to be evaluated.
        mask_file: The masking file.
        patch_file: The file of the adversarial patch.
        model: The targeted local feature extractor.
        device: The computational device.

    Returns:
        mismatch_rate: The matching ratio from the source mask in the source
        image to the target mask in the target image.
    """
    with open(os.path.join(dir, mask_file)) as f:
        mask_datas = json.load(f)

    patch = v2.functional.to_image(
        cv2.imread(patch_file, cv2.IMREAD_GRAYSCALE))
    patch = v2.functional.to_dtype(patch, dtype=torch.float32, scale=True)

    mismtach_rates = []
    for mask_data in mask_datas:
        H = torch.Tensor(mask_data['H'])
        source_view_source_mask = torch.Tensor(mask_data['source_patch'])
        source_view_target_mask = torch.Tensor(mask_data['target_patch'])
        target_view_source_mask = htfm.point_transform(
            H, source_view_source_mask)
        target_view_target_mask = htfm.point_transform(
            H, source_view_target_mask)

        source_view = v2.functional.to_image(
            cv2.imread(os.path.join(dir, mask_data['source_view'])))
        source_view = v2.functional.to_dtype(
            source_view, dtype=torch.float32, scale=True)
        source_view = fill_patch(source_view, patch, source_view_source_mask)
        source_view = fill_patch(source_view, patch, source_view_target_mask)

        target_view = v2.functional.to_image(
            cv2.imread(os.path.join(dir, mask_data['target_view'])))
        target_view = v2.functional.to_dtype(
            target_view, dtype=torch.float32, scale=True)
        target_view = fill_patch(target_view, patch, target_view_source_mask)
        target_view = fill_patch(target_view, patch, target_view_target_mask)

        mismtach_rate = instance_eval(source_view, source_view_source_mask,
                                      target_view, target_view_target_mask,
                                      model, device)
        mismtach_rates.append(mismtach_rate)
    return np.mean(mismtach_rates)


def dirs_eval(dirs: list[str], mask_file: str, patch_file: str, model: Any, device: str = 'cpu'):
    """
    Evaluates every instance with `instance_eval` in a list of directory.

    Args:
        dirs: A list of the directories to be evaluated.
        mask_file: The masking file.
        patch_file: The file of the adversarial patch.
        model: The targeted local feature extractor.
        device: The computational device.
        matrics: A list of metric to evaluate.

    Returns:
        mismatch_rate: The matching ratio from the source mask in the source
        image to the target mask in the target image.
    """
    mismatch_rates = []
    for dir in dirs:
        mismatch_rates.append(dir_eval(dir, mask_file, patch_file,
                                       model, device))
    return np.mean(mismatch_rates)


def main(args):
    if args.model == 'superpoint':
        model = SuperPointNet()
        state_dict = torch.load('models/superpoint_v1.pth')
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError

    if args.dirs:
        mismatch_rate = dirs_eval(args.dirs, args.mask_file,
                                  args.patch_file, model, args.device)
    else:
        mismatch_rate = dir_eval(args.dir, args.mask_file,
                                 args.patch_file, model, args.device)
    print(f'Mismatch-rate: {mismatch_rate}')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dirs', nargs='*')
    parser.add_argument('--dir')
    parser.add_argument('--mask-file', default='mask.json')
    parser.add_argument('--patch-file', default='patch.png')
    parser.add_argument('--model', default='superpoint')
    parser.add_argument('--model-weight', default='models/superpoint_v1.pth')
    parser.add_argument('--device', default='cpu')

    main(parser.parse_args())
