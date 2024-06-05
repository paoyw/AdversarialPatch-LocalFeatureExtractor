from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from tqdm import trange

from models import SuperPointNet
from random import random


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--multiplier", type=float, default=0)
    parser.add_argument("--save", type=str, default="./patch.png")
    parser.add_argument("--model", type=str,
                        default="models/superpoint_v1.pth")
    parser.add_argument("--untargeted", action="store_true")
    parser.add_argument("--init", default="gray")
    parser.add_argument("--init-pattern")
    parser.add_argument("--prob", type=float, default=0.5)
    return parser.parse_args()


transform_pil = v2.Compose([
    v2.ToPILImage()
])


def random_aug(img, args):
    if random() > args.prob:
        return img

    rotate = v2.Compose([
        v2.RandomRotation(180, InterpolationMode.BILINEAR),
        v2.RandomCrop(size=(args.height, args.width)),
    ])

    crop = v2.Compose([
        v2.RandomResizedCrop(size=(args.height, args.width), scale=(0.25, 1)),
    ])

    resize = v2.RandomResize(min_size=args.width // 2, max_size=2 * args.width)

    flip = v2.Compose([
        v2.RandomHorizontalFlip(p=1),
    ])

    augmentations = [crop, resize, flip]
    augmentation = v2.RandomChoice(augmentations)
    return augmentation(img)


def main(args):
    # init args
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    model = SuperPointNet()
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    w, h, pw, ph = args.width, args.height, args.width // 8, args.height // 8
    if args.untargeted:
        target_point = 64
    else:
        target_point = 0
    target = torch.ones((ph * pw), dtype=torch.long) * target_point
    origin = torch.zeros((ph * pw))
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # random initialize
    if args.init_pattern:
        # TODO
        patch = torchvision.io.read_image(args.init_pattern,
                                          torchvision.io.ImageReadMode.GRAY).to(torch.float32)
        patch = (patch / 255).squeeze()
    elif args.init == 'gray':
        patch = torch.ones((h, w), requires_grad=True) * 0.5
    elif args.init == 'rand':
        patch = torch.rand((h, w), requires_grad=True)

    # H x W -> 1 x 1 x H x W
    patch = patch.unsqueeze(0)
    patch = patch.unsqueeze(0)

    model = model.to(device)
    target = target.to(device)
    origin = origin.to(device)
    g = 0

    tbar = trange(args.epoch)
    for _ in tbar:
        patch = patch.detach().clone().to(device)
        patch.requires_grad = True
        sub_patch = random_aug(patch, args)
        model.eval()

        # 1 x 65 x ph x pw, 1 x 256 x ph x pw
        semi, desc = model(sub_patch)
        # 1 x 65 x ph x pw -> 65 x ph x pw
        # 1 x 256 x ph x pw -> 256 x ph x pw
        semi, desc = torch.squeeze(semi, 0), torch.squeeze(desc, 0)
        # 65 x ph x pw -> 65 x (ph x pw)
        # 256 x ph x pw -> 256 x (ph x pw)
        semi, desc = torch.flatten(
            semi, start_dim=1), torch.flatten(desc, start_dim=1)
        # 65 x (ph x pw) -> (ph x pw) x 65
        semi = torch.transpose(semi, 0, 1)
        # 256 x (ph x pw) -> (ph x pw)
        center = desc.mean(dim=0)

        if args.untargeted:
            target_point = 64
        else:
            target_point = 0
        target = torch.ones(semi.shape[:-1], dtype=torch.long) * target_point
        target = target.to(device)
        origin = torch.zeros_like(desc[0])

        # loss and accuracy
        accuracy = (semi.argmax(dim=-1) == target_point).sum() / semi.shape[0]

        loss_mse, loss_ce = 0, 0
        for embed in desc:
            loss_mse = loss_mse + mse_loss(embed - center, origin)
        loss_ce = ce_loss(semi, target)

        tbar.set_description(
            desc=f'mse_loss={loss_mse.item():.3f}, ce_loss={loss_ce.item():.3f}, acc={accuracy:.3f}')

        loss = args.multiplier * loss_mse + loss_ce
        if args.untargeted:
            loss = -loss
        loss.backward()

        # ascent
        grad = patch.grad.detach()
        g = args.decay * g + (grad / torch.norm(grad, p=2))
        # descent
        patch = patch - args.alpha * g

        # clamp
        patch = torch.clamp(patch, 0, 1)

    # 1 x 1 x H x W -> 1 x H x W
    patch = torch.squeeze(patch, 0)
    # 1 x H x W -> H x W
    patch = torch.squeeze(patch, 0)
    img = transform_pil(patch)
    img.save(args.save)


if __name__ == "__main__":
    args = parse_args()
    main(args)
