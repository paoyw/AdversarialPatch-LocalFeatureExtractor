import torch
import torch.nn as nn
from torchvision.transforms import transforms
from argparse import ArgumentParser
from models import SuperPointNet
from tqdm import trange


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--save", type=str, default="./patch.png")
    return parser.parse_args()

transform_pil = transforms.Compose([
    transforms.ToPILImage()
])

def main(args):
    # init args
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    model = SuperPointNet()
    w, h, pw, ph = args.width, args.height, args.width // 8, args.height // 8
    target = torch.ones((ph * pw), dtype=torch.long) * 64
    loss_fn = nn.CrossEntropyLoss()

    # random initialize
    patch = torch.randn((h, w), requires_grad=True)
    # H x W -> 1 x 1 x H x W
    patch = patch.unsqueeze(0)
    patch = patch.unsqueeze(0)

    model = model.to(device)
    target = target.to(device)
    g = 0

    tbar = trange(args.epoch)
    for _ in tbar:
        patch = patch.detach().clone().to(device)
        patch.requires_grad = True
        model.eval()

        # 1 x 65 x ph x pw, 1 x 256 x ph x pw
        semi, _ = model(patch)
        # 1 x 65 x ph x pw -> 65 x ph x pw
        semi = torch.squeeze(semi, 0)
        # 65 x ph x pw -> 65 x (ph x pw)
        semi = torch.flatten(semi, start_dim=1)
        # 65 x (ph x pw) -> (ph x pw) x 65
        semi = torch.transpose(semi, 0, 1)

        # loss and accuracy
        accuracy = (semi.argmax(dim=-1) == 64).sum()
        loss = loss_fn(semi, target)
        tbar.set_description(desc=f'loss={loss.item():.3f}, acc={accuracy/(pw*ph):.3f}')
        loss.backward()

        # ascent
        grad = patch.grad.detach()
        g = args.decay * g + (grad / torch.norm(grad, p=2))
        patch = patch + args.alpha * g

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

