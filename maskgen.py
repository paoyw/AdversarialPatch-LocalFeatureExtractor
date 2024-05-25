from argparse import ArgumentParser
import json
import os
import random

import cv2
import torch

import homography_transforms as htfm


def readHomography(fname: str) -> list[list[float]]:
    """Reads the homography matrix from a file."""
    with open(fname, 'r') as f:
        H = list(map(lambda x: list(map(float, x.split())), f.readlines()))
    return H


def generate_rect(w, h, patch_w, patch_h, method='center') -> list[list[int]]:
    """
    Generates a list of positions of the points.

    Args:
        w: The width of the image.
        h: the height of the image.
        patch_w: The width of the patch.
        patch_h: The height of the patch.
        method: The place for the rectangle, 'center' or 'random'. Default is 'center'.

    Returns:
        A list of positions, whose shape is [4, 2].
        The order is the clock-wise from the top-left.
    """
    if method == 'center':
        x = w // 2 - patch_w // 2
        y = h // 2 - patch_h // 2
    elif method == 'random':
        x = random.randint(0, w - patch_w - 1)
        y = random.randint(0, h - patch_h - 1)
    else:
        raise NotImplementedError
    return [[x, y],
            [x + patch_w, y],
            [x + patch_w, y + patch_h],
            [x, y + patch_h]]


def is_overlapping(rect0: torch.Tensor, rect1: torch.Tensor) -> bool:
    """
    Checks if two rectangles are overlapped with each other.

    Args:
        rect0: The position of each point of a rectangles with order.
        rect1: The position of each point of a rectangles with order.

    Returns:
        If two rectangels are overlapped.
    """
    for p in rect1:
        if p[0] >= rect0[0][0] and p[0] <= rect0[2][0] and \
                p[1] >= rect0[0][1] and p[1] <= rect0[2][1]:
            return True
    return False


def minimum_movement(rect0: torch.Tensor, rect1: torch.Tensor) -> list[int]:
    """
    Compute the minimum Manhaton distance to move one rectangle, so that two
    rectangles is not overlapped.

    Args:
        rect0: The position of each point of a rectangles with order.
        rect1: The position of each point of a rectangles to be moved with order.

    Returns:
        [x, y], the vector of the movements.
    """
    if is_overlapping(rect0, rect1):
        if abs(rect0[2][0] + 1 - rect1[0][0]) < abs(rect0[1][0] - 1 - rect1[2][0]):
            x = rect0[2][0] + 1 - rect1[0][0]
        else:
            x = rect0[1][0] - 1 - rect1[2][0]

        if abs(rect0[2][1] + 1 - rect1[0][1]) < abs(rect0[1][1] - 1 - rect1[2][1]):
            y = rect0[2][1] + 1 - rect1[0][1]
        else:
            y = rect0[1][1] - 1 - rect1[2][1]
        return [int(x), int(y)]
    else:
        return [0, 0]


def generate_mask(dir: str,
                  mask_file: str,
                  patch_width: int | float,
                  patch_height: int | float,
                  H: list | None = None,
                  individual: bool = False,
                  overlapping: bool = False,) -> None:
    """
    Generates the mask of the adversarial patch for each picture in the directory.

    Args:
        dir: The directory of the datas.
        patch_width: The width of the patch, `int` means the absolute width,
                    `float` means the relative width.
        patch_height: The height of the patch, `int` means the absolute height,
                    `float` means the relative height.
        H: The specified homography transformation matrix. The argument
            `individual` will be ignored and set as `True`, if this is used.
            The default value is `None`.
        individual: Uses the homography transformation for each picture. The
                    default value is `False`.
        overlapping: Enable overlapping between two patches. The default value
                     is `False`.

    Returns:
        `None`
    """
    result = []

    img1 = cv2.imread(os.path.join(dir, '1.ppm'))
    h, w, _ = img1.shape

    if isinstance(patch_width, float):
        patch_width = int(w * patch_width)

    if isinstance(patch_height, float):
        patch_height = int(h * patch_height)

    # Computes the homography transformation matrix for each image.
    source_patches = [generate_rect(w, h, patch_width, patch_height)
                      for _ in range(2, 7)]
    if H:
        Hs = [H for _ in range(2, 7)]
    elif individual:
        Hs = [readHomography(os.path.join(dir, f'H_1_{i}'))
              for i in range(2, 7)]
    else:
        i = random.randint(2, 6)
        Hs = [readHomography(os.path.join(dir, f'H_1_{i}'))
              for _ in range(2, 7)]
    target_patches = [
        htfm.point_transform(torch.linalg.inv(torch.Tensor(_H)),
                             torch.Tensor(source_patch)).tolist()
        for _H, source_patch in zip(Hs, source_patches)
    ]

    if not overlapping:
        for i, (source_patch, target_patch) \
                in enumerate(zip(source_patches, target_patches)):
            source_patch = torch.Tensor(source_patch)
            target_patch = torch.Tensor(target_patch)
            if is_overlapping(source_patch, target_patch):
                x, y = minimum_movement(source_patch, target_patch)
                target_patch[:, 0] += x
                target_patch[:, 1] += y
                target_patches[i] = target_patch.tolist()

    for i, _H, source_patch, target_patch in \
            zip(range(2, 7), Hs, source_patches, target_patches):
        result.append({
            'source_view': '1.ppm',
            'target_view': f'{i}.ppm',
            'H': _H,
            'source_patch': source_patch,
            'target_patch': target_patch,
        })

    with open(os.path.join(dir, mask_file), 'w') as f:
        f.write(json.dumps(result, indent=2))


def main(args):
    random.seed(0)

    if args.dirs:
        for dir in args.dirs:
            generate_mask(dir, args.mask_file,
                          args.patch_width, args.patch_height,
                          args.H, args.individual, args.overlapping)
    else:
        generate_mask(args.dir, args.mask_file,
                      args.patch_width, args.patch_height,
                      args.H, args.individual, args.overlapping)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dirs', nargs='*')
    parser.add_argument('--dir')
    parser.add_argument('--mask-file', default='mask.json')
    parser.add_argument('--patch-width', default=128, type=eval)
    parser.add_argument('--patch-height', default=128, type=eval)
    parser.add_argument('--H')
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--overlapping', action='store_true')

    main(parser.parse_args())
