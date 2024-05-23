from argparse import ArgumentParser

import cv2
import numpy as np


def chessboard_gen(rect_width, rect_height, width, height):
    result = 255 * np.ones((width, height)).astype(np.uint8)
    for i in range(0, height, rect_height):
        for j in range(0, width, rect_width):
            if (i // rect_height + j // rect_width) % 2 == 0:
                result = cv2.rectangle(
                    result,
                    [i, j],
                    [i + rect_height - 1, j + rect_width - 1],
                    color=[0, 0, 0],
                    thickness=-1)
    return result


def main(args):
    result = chessboard_gen(rect_width=args.rect_height,
                            rect_height=args.rect_width,
                            width=args.width,
                            height=args.height)
    cv2.imwrite(args.save, result)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--rect_width', default=8, type=int)
    parser.add_argument('--rect_height', default=8, type=int)
    parser.add_argument('--width', default=128, type=int)
    parser.add_argument('--height', default=128, type=int)
    parser.add_argument('--save', default='./patch.png')

    main(parser.parse_args())
