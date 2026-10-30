import os
import argparse
import numpy as np
from PIL import Image
import importlib
import torch

from utils import make_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='CGMRNet')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU 编号，设为 -1 使用 CPU')
    parser.add_argument('--model_path', type=str, default=r'H:\GIT_HUB\CGMRNet\checkpoint\a-light-19\best.pth', help='.pth 权重文件路径')
    parser.add_argument('--indir', default=r'C:\Users\PC\Desktop\data256\2\images', help='输入图片文件夹')
    parser.add_argument('--outdir', default=r'C:\Users\PC\Desktop\data256\CGMR-light2\images', help='预测图片保存文件夹')
    return parser.parse_args()


def main():
    opt = parse_args()
    model_path = os.path.expanduser(opt.model_path)
    if not os.path.isabs(model_path):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            model_path
        )

    model_path = os.path.normpath(model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"权重文件不存在：{model_path}")

    opt.indir = os.path.expanduser(opt.indir)
    opt.outdir = os.path.expanduser(opt.outdir)

    if not os.path.isdir(opt.indir):
        raise FileNotFoundError(f"输入文件夹不存在：{opt.indir}")

    os.makedirs(opt.outdir, exist_ok=True)

    if opt.gpu_id >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 不可用。如需使用 CPU，请设置 --gpu_id -1"
            )

        torch.cuda.set_device(opt.gpu_id)
        device = torch.device(f'cuda:{opt.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    if opt.network != 'CGMRNet':
        raise ValueError(f"不支持的网络结构：{opt.network}")

    try:
        network_module = importlib.import_module(opt.network)
    except ImportError:
        print(f"无法导入网络模块：{opt.network}.py")
        raise

    CGMRNet = getattr(network_module, 'CGMRNet')

    net = CGMRNet(
        in_c=3,
        out_c=3,
        only_residual=False
    ).to(device)

    print(f"Loading weights from: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if 'net_state' in checkpoint:
        state_dict = checkpoint['net_state']
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }

    net.load_state_dict(state_dict)
    net.eval()

    test_img_paths = make_dataset(opt.indir)

    if len(test_img_paths) == 0:
        raise RuntimeError(f"输入文件夹中没有找到图片：{opt.indir}")

    print(f"Found {len(test_img_paths)} images.")
    print("Start predicting...")

    with torch.no_grad():
        for index, img_path in enumerate(test_img_paths, start=1):
            with Image.open(img_path) as source_img:
                img = source_img.convert('RGB')

            im_w, im_h = img.size

            if im_w % 4 != 0 or im_h % 4 != 0:
                new_w = im_w // 4 * 4
                new_h = im_h // 4 * 4

                if new_w == 0 or new_h == 0:
                    raise ValueError(f"图片尺寸过小：{img_path}")

                img = img.resize((new_w, new_h))

            img_array = np.asarray(img, dtype=np.float32)

            img_data = torch.from_numpy(
                img_array.transpose(2, 0, 1).copy()
            )

            in_data = (img_data.unsqueeze(0) - 128.0).to(device)

            pred = net(in_data)

            out_img_data = (
                pred[0]
                .cpu()
                .float()
                .round()
                .clamp(0, 255)
            )

            out_array = (
                out_img_data
                .numpy()
                .astype(np.uint8)
                .transpose(1, 2, 0)
            )

            out_img = Image.fromarray(out_array)


            filename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(opt.outdir, filename + '.png')
            out_img.save(save_path)

            print(
                f"[{index}/{len(test_img_paths)}] "
                f"Saved: {save_path}"
            )

    print(f"\n预测完成，结果保存在：{opt.outdir}")


if __name__ == '__main__':
    main()