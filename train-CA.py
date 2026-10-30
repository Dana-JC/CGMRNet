import os
import datetime
import time 
import argparse
import numpy as np
import random 
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ImagePairPrefixFolder import ImagePairPrefixFolder, var_custom_collate ,test_collate
from utils import MovingAvg, ssim
from tf_visualizer import TFVisualizer
import importlib 
from lossfunction import MSEEdgeLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='CGMRNet')
parser.add_argument('--networkname', default='CGMRNet')
parser.add_argument('--name', default='')
parser.add_argument('--seed', default='42')
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)   
parser.add_argument('--lr_min', type=float, default=5e-6)
parser.add_argument('--weight_decay', type=float, default=0.009)
parser.add_argument('--alpha_start',type=float,default=0.05)
parser.add_argument('--alpha_end',type=float,default=0)
parser.add_argument('--alpha_T',type=float,default=500)
parser.add_argument('--crop_size',type=int,default=256,help='Training crop size')
parser.add_argument('--checkpoints_dir', default='checkpoint')
parser.add_argument('--logDir', default='tblogdir')
parser.add_argument('--pretrained',default='',type=str,help='path to pretrained model') 
parser.add_argument('--freeze_encoder',action='store_true',help='freeze encoder layers') 
parser.add_argument('--resume_dir', default='')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--save_epoch', type=int, default=20)
parser.add_argument('--save_latest_freq', type=int, default=5000)
parser.add_argument('--test_epoch', type=int, default=10) 
parser.add_argument('--test_max_size', type=int, default=1080)
parser.add_argument('--size_unit', type=int,  default=8)
parser.add_argument('--print_iter', type=int,  default=100)
parser.add_argument('--input_folder', default='H:\\IMDITE\\light_fog\\train\\hz')
parser.add_argument('--gt_folder', default='H:\\IMDITE\\light_fog\\train\\gt')
parser.add_argument('--test_input_folder', default='H:\\IMDITE\\light_fog\\val\\hz')
parser.add_argument('--test_gt_folder', default='H:\\IMDITE\\light_fog\\val\\gt')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--only_residual', action='store_true', help='regress residual rather than image') 
parser.add_argument('--loss_func', default='mse_edge', help='l2|l1|mse_freq')
parser.add_argument('--inc', type=int, default=3)
parser.add_argument('--outc', type=int, default=3)
parser.add_argument('--force_rgb', action='store_true') 
parser.add_argument('--no_edge', type=bool, default=True, help='True: 3-channel RGB, False: 4-channel RGB+Edge')
parser.add_argument('--optimizer', default='adam', type=str, help='adam | sgd | adamw | rmsprop')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD or RMSprop')

opt = parser.parse_args()

if __name__ == '__main__':

    set_seed(int(opt.seed))  

    opt.input_folder = os.path.expanduser(opt.input_folder)
    opt.gt_folder = os.path.expanduser(opt.gt_folder)
    opt.test_input_folder = os.path.expanduser(opt.test_input_folder)
    opt.test_gt_folder = os.path.expanduser(opt.test_gt_folder)

    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
        os.makedirs(os.path.join(opt.checkpoints_dir, opt.name))
    opt.resume_dir = opt.resume_dir if opt.resume_dir != '' else os.path.join(opt.checkpoints_dir, opt.name)

    visualizer = TFVisualizer(opt)
    # ### Log out
    # with open(os.path.realpath(__file__), 'r', encoding='utf-8') as fid:
    #     visualizer.print_logs(fid.read())

    ## print argument
    for key, val in vars(opt).items():
        visualizer.print_logs('%s: %s' % (key, val))

    opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(',')]
    assert all(0 <= x <= torch.cuda.device_count() for x in opt.gpu_ids), 'gpu id should ' \
                                                                          'be 0~{0}'.format(torch.cuda.device_count())
    torch.cuda.set_device(opt.gpu_ids[0])


    train_dataset = ImagePairPrefixFolder(opt.input_folder, opt.gt_folder, size_unit=opt.size_unit, force_rgb=opt.force_rgb , crop_size=opt.crop_size)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  collate_fn=var_custom_collate, pin_memory=True,
                                  num_workers=opt.num_workers)

    opt.do_test = opt.test_gt_folder != ''
    if opt.do_test:
        test_dataset = ImagePairPrefixFolder(opt.test_input_folder, opt.test_gt_folder,
                                             max_img_size=opt.test_max_size, size_unit=opt.size_unit, force_rgb=opt.force_rgb)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     collate_fn=test_collate , pin_memory=True,
                                     num_workers=1)

    total_inc = opt.inc if opt.no_edge else opt.inc + 1

    if opt.network == 'CGMRNet':
        try:
            network_module = importlib.import_module(opt.networkname)
        except ImportError:
            print(f"Error: Cannot find module {opt.networkname}.py")
            raise
        CGMRNet = getattr(network_module, 'CGMRNet') 
        net = CGMRNet(in_c=total_inc, out_c=3, only_residual=opt.only_residual)
                # ============================
        # Load pretrained model
        # ============================
        if opt.pretrained != '':
        
            print(
                'Loading pretrained model:',
                opt.pretrained
            )
        
            checkpoint = torch.load(
                opt.pretrained,
                map_location='cpu'
            )
        
        
            if 'net_state' in checkpoint:
                pretrained_dict = checkpoint['net_state']
            else:
                pretrained_dict = checkpoint
        
        
            model_dict = net.state_dict()
        
        
            # 过滤不匹配参数
            pretrained_dict = {
                k:v for k,v in pretrained_dict.items()
                if k in model_dict 
                and v.size()==model_dict[k].size()
            }
        
        
            model_dict.update(pretrained_dict)
        
            net.load_state_dict(model_dict)
        
            print(
                'Loaded parameters:',
                len(pretrained_dict),
                '/',
                len(model_dict)
            )
    else:
        print('network structure %s not supported' % opt.networkname)
        raise ValueError
    # ==========================================
    # 新增：计算并打印模型参数量
    # ==========================================
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    visualizer.print_logs('===============================================')
    visualizer.print_logs(f'Total parameters: {total_params:,}')
    visualizer.print_logs(f'Trainable parameters: {trainable_params:,}')
    visualizer.print_logs('===============================================')


    if opt.loss_func == 'l2':
        loss_crit = torch.nn.MSELoss()

    elif opt.loss_func == 'l1':
        loss_crit = torch.nn.SmoothL1Loss()

    elif opt.loss_func == 'mse_edge':
        loss_crit=MSEEdgeLoss(
            alpha=opt.alpha_start
        )


    else:
        raise ValueError(
            'loss_func %s not supported'
            % opt.loss_func
        )
    
    pnsr_crit = torch.nn.MSELoss()

    if len(opt.gpu_ids) > 0:
        net.cuda()
        if len(opt.gpu_ids) > 1:
            net = torch.nn.DataParallel(net)
        loss_crit = loss_crit.cuda()
        pnsr_crit = pnsr_crit.cuda()

# ==========================================
    # 修改：根据配置动态选择优化器
    # ==========================================
    opt_name = opt.optimizer.lower()
    if opt_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt_name == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise ValueError(f"Optimizer '{opt.optimizer}' is not supported. Please choose from: adam, adamw, sgd, rmsprop.")
        
    step_optim_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=opt.epochs,
    eta_min=opt.lr_min)
    loss_avg = MovingAvg(pool_size=50)

    start_epoch = 0
    total_iter = 0
    # Best model
    best_psnr = -1.0
    best_ssim = -1.0
    best_epoch = -1 

    if os.path.exists(os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth')):
        print('resuming from latest.pth')
        latest_info = torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth'))
        start_epoch = latest_info['epoch'] + 1
        total_iter = latest_info['total_iter']
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(latest_info['net_state'])
        else:
            net.load_state_dict(latest_info['net_state'])
        optimizer.load_state_dict(latest_info['optim_state'])
        visualizer.print_logs(
          "Resume LR: %.8f" %
          optimizer.param_groups[0]['lr']
        )
        if 'scheduler_state' in latest_info:
            step_optim_scheduler.load_state_dict(
              latest_info['scheduler_state']
            )
        best_psnr = latest_info.get('best_psnr', -1.0)
        best_ssim = latest_info.get('best_ssim', -1.0)
        best_epoch = latest_info.get('best_epoch', -1)

    if opt.resume_epoch > 0:
        start_epoch = opt.resume_epoch
        total_iter = opt.resume_epoch * len(train_dataloader)
        resume_path = os.path.join(opt.resume_epoch, 'net_epoch_%d.pth') % opt.resume_epoch
        print('resume from : %s' % resume_path)
        assert os.path.exists(resume_path), 'cannot find the resume model: %s ' % resume_path
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(torch.load(resume_path))
        else:
            net.load_state_dict(torch.load(resume_path))

    for epoch in range(start_epoch, opt.epochs):

        def cosine_alpha(epoch, max_epoch, alpha_min=0.03, alpha_max=0.06):
            alpha = alpha_min + (
                alpha_max - alpha_min
            ) * (1 - math.cos(math.pi * epoch / max_epoch)) / 2
            return alpha
        
        epoch_start_time = time.time()
        current_alpha = cosine_alpha(
            epoch,
            opt.alpha_T,
            opt.alpha_start,
            opt.alpha_end
        )
        loss_crit.update_alpha(current_alpha)
        visualizer.print_logs( "Current alpha: %.5f" % current_alpha)
        visualizer.print_logs("Start to train epoch %d" % epoch)
        net.train()
        for iter, data in enumerate(train_dataloader):
            total_iter += 1
            optimizer.zero_grad()

            batch_input_img, batch_input_edge,  batch_gt = data
            if len(opt.gpu_ids) > 0:
                batch_input_img, batch_input_edge, batch_gt = batch_input_img.cuda(), batch_input_edge.cuda(), batch_gt.cuda()

            if opt.no_edge:
                batch_input = batch_input_img
            else:
                batch_input = torch.cat((batch_input_img, batch_input_edge), dim=1)
            batch_input_v = Variable(batch_input)
            if opt.only_residual:
                batch_gt_v = Variable(batch_gt - (batch_input_img+128))
            else:
                batch_gt_v = Variable(batch_gt)

            pred = net(batch_input_v)

            loss = loss_crit(pred, batch_gt_v)
            avg_loss = loss_avg.set_curr_val(loss.data)

            loss.backward()
            optimizer.step()

            if iter % opt.print_iter == 0:
                visualizer.plot_current_losses(total_iter, { 'loss': loss})
                visualizer.print_logs('%s Step[%d/%d], lr: %f, mv_avg_loss: %f, loss: %f' %
                                        (str(datetime.datetime.now()).split(' ')[1], iter, len(train_dataloader),
                                         optimizer.param_groups[0]['lr'], avg_loss, loss))

            if total_iter % opt.save_latest_freq == 0:
                latest_info = {
                                'total_iter': total_iter,
                                'epoch': epoch + 1,                     # 下一轮开始的epoch
                                'optim_state': optimizer.state_dict(),
                                'scheduler_state': step_optim_scheduler.state_dict(),   # 新增
                                'best_psnr': best_psnr,
                                'best_ssim': best_ssim,
                                'best_epoch': best_epoch,
                            }
                if len(opt.gpu_ids) > 1:
                    latest_info['net_state'] = net.module.state_dict()
                else:
                    latest_info['net_state'] = net.state_dict()
                print('save lastest model.')
                torch.save(latest_info, os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth'))

        step_optim_scheduler.step()
        latest_info = {
            'total_iter': total_iter,
            'epoch': epoch + 1,
            'optim_state': optimizer.state_dict(),
            'scheduler_state': step_optim_scheduler.state_dict(),
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'best_epoch': best_epoch,
        }
        
        if len(opt.gpu_ids) > 1:
            latest_info['net_state'] = net.module.state_dict()
        else:
            latest_info['net_state'] = net.state_dict()
        
        torch.save(
            latest_info,
            os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth')
        )

        if (epoch+1) % opt.save_epoch == 0 :
            visualizer.print_logs('saving model for epoch %d' % epoch)
            if len(opt.gpu_ids) > 1:
                torch.save(net.module.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'net_epoch_%d.pth' % (epoch+1)))
            else:
                torch.save(net.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'net_epoch_%d.pth' % (epoch + 1)))

        if opt.do_test and (epoch + 1) % opt.test_epoch == 0:        
            avg_psnr = 0
            avg_ssim = 0
            task_cnt = 0
            net.eval()
            with torch.no_grad():
                for iter, data in enumerate(test_dataloader):
                    batch_input_img, batch_input_edge,  batch_gt = data
                    if len(opt.gpu_ids) > 0:
                        batch_input_img, batch_input_edge, batch_gt = batch_input_img.cuda(), batch_input_edge.cuda(), batch_gt.cuda()

                    if opt.no_edge:
                        batch_input = batch_input_img
                    else:
                        batch_input = torch.cat((batch_input_img, batch_input_edge), dim=1)
                    batch_input_v = Variable(batch_input)
                    batch_gt_v = Variable(batch_gt)


                    pred = net(batch_input_v)

                    if opt.only_residual:
                        pred_img = pred + Variable(batch_input_img+128)
                        loss = pnsr_crit(pred_img, batch_gt_v)
                    else:
                        pred_img = pred
                        loss = pnsr_crit(pred, batch_gt_v)
                    avg_psnr += 10 * np.log10(255 * 255 / loss.item())
                    
                    pred_numpy = pred_img[0].cpu().numpy().transpose(1, 2, 0).astype(np.float64)
                    gt_numpy = batch_gt[0].cpu().numpy().transpose(1, 2, 0).astype(np.float64)
                    
                    if pred_numpy.shape[2] == 3:
                        ssim_val = 0
                        for c in range(3):
                            ssim_val += ssim(pred_numpy[:,:,c], gt_numpy[:,:,c])
                        avg_ssim += ssim_val / 3
                    else:
                        avg_ssim += ssim(pred_numpy, gt_numpy)
                    task_cnt += 1

            visualizer.print_logs('Testing for epoch: %d' % epoch)
            visualizer.print_logs('Average test PSNR is %f for %d images' % (avg_psnr/task_cnt, task_cnt))
            visualizer.print_logs('Average test SSIM is %f for %d images' % (avg_ssim/task_cnt, task_cnt))
            current_psnr = avg_psnr / task_cnt
            current_ssim = avg_ssim / task_cnt

            if current_psnr >= best_psnr:

                best_psnr = current_psnr
                best_ssim = current_ssim
                best_epoch = epoch + 1

                best_info = {
                    'epoch': best_epoch,
                    'psnr': best_psnr,
                    'ssim': best_ssim,
                    'net_state': net.module.state_dict() if len(opt.gpu_ids) > 1 else net.state_dict()
                }

                torch.save(
                    best_info,
                    os.path.join(opt.checkpoints_dir, opt.name, 'best.pth')
                )

                visualizer.print_logs(
                    '====== Best model updated ======'
                )
                visualizer.print_logs(
                    'Best Epoch : %d' % best_epoch
                )
                visualizer.print_logs(
                    'Best PSNR  : %.4f' % best_psnr
                )
                visualizer.print_logs(
                    'Best SSIM  : %.6f' % best_ssim
                )
        # ==========================================
        # 新增：计算并打印当前 Epoch 耗时
        # ==========================================
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        formatted_time = str(datetime.timedelta(seconds=int(epoch_duration)))
        visualizer.print_logs('Epoch %d finished. Time taken: %s' % (epoch, formatted_time))
        visualizer.print_logs('-----------------------------------------------')
    visualizer.print_logs('===========================================')
    visualizer.print_logs('Training Finished!')
    visualizer.print_logs('Best Model Summary')
    visualizer.print_logs('Best Epoch : %d' % best_epoch)
    visualizer.print_logs('Best PSNR  : %.4f' % best_psnr)
    visualizer.print_logs('Best SSIM  : %.6f' % best_ssim)
    visualizer.print_logs(
        'Best Model : %s'
        % os.path.join(opt.checkpoints_dir, opt.name, 'best.pth')
    )
    visualizer.print_logs('===========================================')    