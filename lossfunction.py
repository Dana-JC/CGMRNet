
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()

        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],
            dtype=torch.float32
        )

        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]],
            dtype=torch.float32
        )

        self.register_buffer(
            'weight_x',
            sobel_x.unsqueeze(0)
        )

        self.register_buffer(
            'weight_y',
            sobel_y.unsqueeze(0)
        )

    def rgb2gray(self, x):

        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        return gray

    def forward(self, pred, gt):

        pred_gray = self.rgb2gray(pred)
        gt_gray = self.rgb2gray(gt)

        pred_x = F.conv2d(
            pred_gray,
            self.weight_x,
            padding=1
        )

        pred_y = F.conv2d(
            pred_gray,
            self.weight_y,
            padding=1
        )

        gt_x = F.conv2d(
            gt_gray,
            self.weight_x,
            padding=1
        )

        gt_y = F.conv2d(
            gt_gray,
            self.weight_y,
            padding=1
        )

        edge_loss = (
            F.mse_loss(pred_x, gt_x)
            +
            F.mse_loss(pred_y, gt_y)
        )

        return edge_loss

class MSEEdgeLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super(MSEEdgeLoss, self).__init__()

        self.mse = nn.MSELoss()

        self.edge = EdgeLoss()

        self.alpha = alpha

    def update_alpha(self, alpha):
        self.alpha = alpha


    def forward(self, pred, gt):

        mse_loss = self.mse(pred, gt)

        edge_loss = self.edge(pred, gt)

        loss = mse_loss + self.alpha * edge_loss

        return loss