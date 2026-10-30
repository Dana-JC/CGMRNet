import os
import argparse
import numpy as np
import random
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import importlib
from ImagePairPrefixFolder import ImagePairPrefixFolder, var_custom_collate
from utils import ssim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='CGMRNet')
parser.add_argument('--networkname', default='CGMRNet')
parser.add_argument('--model_path', type=str, default='weight/medium/best.pth', help='待测试的 .pth 权重文件路径')
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--test_max_size', type=int, default=1080)
parser.add_argument('--size_unit', type=int, default=8)
parser.add_argument('--test_input_folder', default=r'H:\IMDITE\medium_fog\val\hz')
parser.add_argument('--test_gt_folder', default=r'H:\IMDITE\medium_fog\val\gt')
parser.add_argument('--only_residual', action='store_true')
parser.add_argument('--inc', type=int, default=3)
parser.add_argument('--outc', type=int, default=3)
parser.add_argument('--force_rgb', action='store_true')
parser.add_argument('--no_edge', type=bool, default=True)

opt = parser.parse_args()


def main():
    opt.test_input_folder = os.path.expanduser(opt.test_input_folder)
    opt.test_gt_folder = os.path.expanduser(opt.test_gt_folder)

    opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(',')]
    torch.cuda.set_device(opt.gpu_ids[0])
    print(f"Using GPU: {opt.gpu_ids[0]}")

    model_path = os.path.expanduser(opt.model_path)

    if not os.path.isabs(model_path):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            model_path
        )

    model_path = os.path.normpath(model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"权重文件不存在：{model_path}")

    print(f"Loading test dataset from: {opt.test_input_folder}")

    test_dataset = ImagePairPrefixFolder(
        opt.test_input_folder,
        opt.test_gt_folder,
        max_img_size=opt.test_max_size,
        size_unit=opt.size_unit,
        force_rgb=opt.force_rgb
    )

    if len(test_dataset) == 0:
        raise RuntimeError("未找到测试图像，请检查输入图像和 GT 路径。")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=var_custom_collate,
        pin_memory=True,
        num_workers=1
    )

    total_inc = opt.inc if opt.no_edge else opt.inc + 1

    try:
        network_module = importlib.import_module(opt.networkname)
    except ImportError:
        print(f"Error: Cannot import module {opt.networkname}.py")
        raise

    CGMRNet = getattr(network_module, 'CGMRNet')
    net = CGMRNet(
        in_c=total_inc,
        out_c=3,
        only_residual=opt.only_residual
    )

    net.cuda()

    if len(opt.gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)

    print(f"Loading weights from: {model_path}")

    state_dict = torch.load(
        model_path,
        map_location=f'cuda:{opt.gpu_ids[0]}'
    )

    if 'net_state' in state_dict:
        state_dict = state_dict['net_state']

    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)

    psnr_crit = torch.nn.MSELoss().cuda()

    avg_psnr = 0.0
    avg_ssim = 0.0
    task_cnt = 0

    net.eval()
    print("Start evaluating...")
    start_time = time.time()

    with torch.no_grad():
        for data in test_dataloader:
            batch_input_img, batch_input_edge, batch_gt = data

            batch_input_img = batch_input_img.cuda()
            batch_input_edge = batch_input_edge.cuda()
            batch_gt = batch_gt.cuda()

            if opt.no_edge:
                batch_input = batch_input_img
            else:
                batch_input = torch.cat(
                    (batch_input_img, batch_input_edge),
                    dim=1
                )

            batch_input_v = Variable(batch_input)
            batch_gt_v = Variable(batch_gt)

            pred = net(batch_input_v)

            loss = psnr_crit(pred, batch_gt_v)
            mse = loss.item()

            current_psnr = (
                float('inf')
                if mse == 0
                else 10 * np.log10(255 * 255 / mse)
            )
            avg_psnr += current_psnr

            pred_numpy = (
                pred[0]
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.float64)
            )

            gt_numpy = (
                batch_gt[0]
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.float64)
            )

            if pred_numpy.shape[2] == 3:
                ssim_val = 0.0

                for c in range(3):
                    ssim_val += ssim(
                        pred_numpy[:, :, c],
                        gt_numpy[:, :, c]
                    )

                current_ssim = ssim_val / 3
            else:
                current_ssim = ssim(pred_numpy, gt_numpy)

            avg_ssim += current_ssim
            task_cnt += 1

            if task_cnt % 5 == 0 or task_cnt == len(test_dataset):
                print(
                    f"Tested [{task_cnt}/{len(test_dataset)}] images..."
                )

    print("\n" + "=" * 47)
    print(f'Testing weights: {model_path}')
    print(
        f'Average test PSNR is {avg_psnr / task_cnt:.6f} '
        f'for {task_cnt} images'
    )
    print(
        f'Average test SSIM is {avg_ssim / task_cnt:.6f} '
        f'for {task_cnt} images'
    )
    print(f'Time taken: {time.time() - start_time:.2f} seconds')
    print("=" * 47)


if __name__ == '__main__':
    main()