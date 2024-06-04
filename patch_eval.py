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
from tqdm import tqdm

import homography_transforms as htfm
from models import SuperPointNet, SIFT


def in_convex_hull(points, hull: ConvexHull) -> bool:
    """
    Checks if the point is in the convex hull.

    Args:
        point: A list of points to be checked.
        hull: The convex hull to be checked.

    Returns:
        Returns if the point is in the convex hull
    """
    hull = Delaunay(hull.points)
    return hull.find_simplex(points) != -1


def interpolation(point, h, w):
    """Computes the nearest four neighbors with the coresponding weights for bilinear interpolation."""
    points = []
    weights = []
    i_min = np.clip(np.floor(point[1]), 0, h - 1)
    i_max = np.clip(np.ceil(point[1]), 0, h - 1)
    j_min = np.clip(np.floor(point[0]), 0, w - 1)
    j_max = np.clip(np.ceil(point[0]), 0, w - 1)
    if i_min == i_max and j_min == j_max:
        points.append([int(j_min), int(i_min)])
        weights.append(1)
    elif i_min == i_max:
        for j in [j_min, j_max]:
            points.append([int(j), int(i_min)])
            weights.append((1 - np.abs(j - point[0])))
    elif j_min == j_max:
        for i in [i_min, i_max]:
            points.append([int(j_min), int(i)])
            weights.append((1 - np.abs(i - point[1])))
    else:
        for i, j in itertools.product([i_min, i_max], [j_min, j_max]):
            points.append([int(j), int(i)])
            weights.append(
                (1 - np.abs(j - point[0])) * (1 - np.abs(i - point[1])))
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

    points = []
    for i in range(i_start, i_end + 1):
        for j in range(j_start, j_end + 1):
            points.append([j, i])
    points = np.array(points)
    points = points[in_convex_hull(points, mask_hull)]
    for j, i in points:
        src_pt = np.array([j, i, 1], dtype=np.float32)
        dst_pt = (H @ src_pt.T).T.squeeze()
        dst_pt = dst_pt[:2] / dst_pt[-1:]
        pts, weights = interpolation(dst_pt, patch_h, patch_w)
        img[:, i, j] = 0
        for pt, weight in zip(pts, weights):
            img[:, i, j] += patch[:, pt[1], pt[0]] * weight
    return img


def calculateRepeatability(img1, img2, H1to2, keypoints1, keypoints2t, threshold: float = 3):
    H2to1 = torch.linalg.inv(H1to2)
    keypoints1t = htfm.point_transform(H1to2, keypoints1)
    keypoints1 = keypoints1[
        torch.where((keypoints1t[:, 0] >= 0) &
                    (keypoints1t[:, 0] < img2.shape[-1]) &
                    (keypoints1t[:, 1] >= 0) &
                    (keypoints1t[:, 1] < img2.shape[-2]))
    ]

    keypoints2 = htfm.point_transform(H2to1, keypoints2t)
    keypoints2 = keypoints2[
        torch.where((keypoints2[:, 0] >= 0) &
                    (keypoints2[:, 0] < img1.shape[-1]) &
                    (keypoints2[:, 1] >= 0) &
                    (keypoints2[:, 1] < img1.shape[-2]))
    ]

    dist = torch.square(keypoints1[:, None, :] - keypoints2[None, :, :])
    dist = torch.sum(dist, axis=-1)
    dist = torch.sqrt(dist)
    return (dist < threshold).sum().cpu().item() / min(keypoints1.shape[0], keypoints2.shape[0])


def evalHomographyEstimation(h, w, predH, targetH, threshold=1):
    points = torch.Tensor([[0, 0], [0, h], [w, h], [w, 0]])
    pred_point = htfm.point_transform(predH, points)
    target_point = htfm.point_transform(targetH, points)
    dist = (pred_point - target_point).norm(dim=-1)
    return float((dist <= threshold).sum() / len(dist))


def instance_eval(source_view: torch.Tensor,
                  source_view_source_mask: torch.Tensor,
                  source_view_target_mask: torch.Tensor,
                  target_view: torch.Tensor,
                  target_view_source_mask: torch.Tensor,
                  target_view_target_mask: torch.Tensor,
                  H: torch.Tensor,
                  model: Any, device: str = 'cpu') -> dict:
    """
    Evaluates an instance of an adversarial patch attack for the local feature extractor.

    Args:
        source_view: The image from the source view.
        source_view_source_mask: The source mask in the `source_view` image.
        source_view_target_mask: The target mask in the `source_view` image.
        target_view: The image from the target view.
        target_view_source_mask: The source mask in the `target_view` image.
        target_view_target_mask: The target mask in the `target_view` image.
        H: The ground truth of the homography.
        model: The targeted local feature extractor.
        device: The computational device.

    Returns:
        result: A dictionary of the result.
    """
    source_view = source_view.to(device).unsqueeze(dim=0)
    target_view = target_view.to(device).unsqueeze(dim=0)

    source_view_source_hull = ConvexHull(source_view_source_mask)
    source_view_target_hull = ConvexHull(source_view_target_mask)
    target_view_source_hull = ConvexHull(target_view_source_mask)
    target_view_target_hull = ConvexHull(target_view_target_mask)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        source_view_pt, source_view_desc = model.to_cv2(
            v2.functional.rgb_to_grayscale(source_view))
        target_view_pt, target_view_desc = model.to_cv2(
            v2.functional.rgb_to_grayscale(target_view))
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(source_view_desc, target_view_desc, k=2)
    matches = sorted(matches,
                     key=lambda m: np.abs(m[0].distance - m[1].distance))
    matches = matches

    source_total = 0
    target_total = 0
    t_count = 0
    f_count = 0
    for m, _ in matches[:1000]:
        if in_convex_hull(source_view_pt[m.queryIdx].pt, source_view_source_hull):
            source_total += 1
            if in_convex_hull(target_view_pt[m.trainIdx].pt, target_view_source_hull):
                t_count += 1
            if in_convex_hull(target_view_pt[m.trainIdx].pt, target_view_target_hull):
                f_count += 1
        if in_convex_hull(source_view_pt[m.queryIdx].pt, source_view_target_hull):
            target_total += 1

    filtered_source_pt = np.float32([
        source_view_pt[m.queryIdx].pt for m, _ in matches[:1000]
    ])
    filtered_target_pt = np.float32([
        target_view_pt[m.trainIdx].pt for m, _ in matches[:1000]
    ])

    Hs2t, Hmask = cv2.findHomography(
        filtered_source_pt, filtered_target_pt, cv2.RANSAC
    )
    Hs2t = torch.Tensor(Hs2t)

    if filtered_source_pt.shape[0] <= 0 or filtered_target_pt.shape[0] <= 0:
        rep = 0
    else:
        rep = calculateRepeatability(
            source_view.squeeze(),
            target_view.squeeze(),
            H.to(device),
            torch.Tensor(filtered_source_pt).to(device),
            torch.Tensor(filtered_target_pt).to(device),
        )

    he1 = evalHomographyEstimation(h=source_view.shape[-2],
                                   w=source_view.shape[-1],
                                   predH=Hs2t,
                                   targetH=H,
                                   threshold=1)
    he3 = evalHomographyEstimation(h=source_view.shape[-2],
                                   w=source_view.shape[-1],
                                   predH=Hs2t,
                                   targetH=H,
                                   threshold=3)
    he5 = evalHomographyEstimation(h=source_view.shape[-2],
                                   w=source_view.shape[-1],
                                   predH=Hs2t,
                                   targetH=H,
                                   threshold=5)

    results = {'total point': len(matches),
               'source total points': source_total,
               'target total points': target_total,
               'source points ratio': source_total / len(matches),
               'target points ratio': target_total / len(matches),
               't_count': t_count,
               'f_count': f_count,
               'TP': t_count / source_total if source_total != 0 else 0,
               'FP': f_count / source_total if source_total != 0 else 0,
               'repeatability': rep,
               'homography estimation 1': he1,
               'homography estimation 3': he3,
               'homography estimation 5': he5, }
    return results


def dir_eval(dir: str, mask_file: str, patch_file: str,
             model: Any, device: str = 'cpu',
             null_mask: bool = False) -> dict:
    """
    Evaluates every instance with `instance_eval` in a directory.

    Args:
        dir: The directory to be evaluated.
        mask_file: The masking file.
        patch_file: The file of the adversarial patch.
        model: The targeted local feature extractor.
        device: The computational device.
        null_mask: Evaluate without filling the mask with the patch.

    Returns:
        results: A dictionary of the result.
    """
    with open(os.path.join(dir, mask_file)) as f:
        mask_datas = json.load(f)

    patch = v2.functional.to_image(
        cv2.imread(patch_file, cv2.IMREAD_GRAYSCALE))
    patch = v2.functional.to_dtype(patch, dtype=torch.float32, scale=True)

    results = {}
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
        if not null_mask:
            source_view = fill_patch(source_view, patch,
                                     source_view_source_mask)
            source_view = fill_patch(source_view, patch,
                                     source_view_target_mask)

        target_view = v2.functional.to_image(
            cv2.imread(os.path.join(dir, mask_data['target_view'])))
        target_view = v2.functional.to_dtype(
            target_view, dtype=torch.float32, scale=True)
        if not null_mask:
            target_view = fill_patch(target_view, patch,
                                     target_view_source_mask)
            target_view = fill_patch(target_view, patch,
                                     target_view_target_mask)

        result = instance_eval(source_view=source_view,
                               source_view_source_mask=source_view_source_mask,
                               source_view_target_mask=source_view_target_mask,
                               target_view=target_view,
                               target_view_source_mask=target_view_source_mask,
                               target_view_target_mask=target_view_target_mask,
                               H=H,
                               model=model, device=device)
        for k, v in result.items():
            if k not in results:
                results[k] = []
            results[k].append(v)
    for k, v in results.items():
        results[k] = np.mean(v)
    return results


def dirs_eval(dirs: list[str],
              mask_file: str,
              patch_file: str,
              model: Any, device: str = 'cpu',
              null_mask: bool = False) -> dict:
    """
    Evaluates every instance with `instance_eval` in a list of directory.

    Args:
        dirs: A list of the directories to be evaluated.
        mask_file: The masking file.
        patch_file: The file of the adversarial patch.
        model: The targeted local feature extractor.
        device: The computational device.
        matrics: A list of metric to evaluate.
        null_mask: Evaluate without filling the mask with the patch.

    Returns:
        results: A dictionary of the result.
    """
    results = {}
    pbar = tqdm(dirs, ncols=50)
    for dir in pbar:
        result = dir_eval(dir=dir,
                          mask_file=mask_file,
                          patch_file=patch_file,
                          model=model,
                          device=device,
                          null_mask=null_mask)
        pbar.write(f'{dir}/{mask_file}: {result}')
        for k, v in result.items():
            if k not in results:
                results[k] = []
            results[k].append(v)
    for k, v in results.items():
        results[k] = np.mean(v)
    return results


def main(args):
    if args.model == 'superpoint':
        model = SuperPointNet()
        state_dict = torch.load('models/superpoint_v1.pth')
        model.load_state_dict(state_dict)
    elif args.model == 'sift':
        model = SIFT()
    else:
        raise NotImplementedError

    if args.dirs:
        result = dirs_eval(dirs=args.dirs,
                           mask_file=args.mask_file,
                           null_mask=args.null_mask,
                           patch_file=args.patch_file,
                           model=model,
                           device=args.device)
    else:
        result = dir_eval(dir=args.dir,
                          mask_file=args.mask_file,
                          null_mask=args.null_mask,
                          patch_file=args.patch_file,
                          model=model,
                          device=args.device)
    print(result)
    result['args'] = vars(args)
    with open(args.log, 'w') as f:
        f.write(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dirs', nargs='*')
    parser.add_argument('--dir')
    parser.add_argument('--mask-file', default='mask.json')
    parser.add_argument('--null-mask', action='store_true')
    parser.add_argument('--patch-file', default='patch.png')
    parser.add_argument('--model', default='superpoint')
    parser.add_argument('--model-weight', default='models/superpoint_v1.pth')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--log', default='log.json')

    main(parser.parse_args())
