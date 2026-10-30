import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class CGAF(nn.Module):
    def __init__(self, dim=64):
        super(CGAF, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim//2, 1, 1, bias=False)
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, y1, y2, y3):
        f1 = F.relu(self.conv1(y3))
        f2 = F.sigmoid(self.conv2(f1))
        f3 = y1 * f2
        f4 = y2 * f2
        f = torch.cat((y3, f3, f4), dim=1)
        out = self.fuse(f)      
        
        return out

    
class LFEB_F(nn.Module):
    def __init__(self, dim):
        super(LFEB_F, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = nn.InstanceNorm2d(dim)
        
        self.pwconv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.act = nn.GELU()
        
        self.pwconv2 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + shortcut
    
class LFEB_T(nn.Module):
    def __init__(self, dim):
        super(LFEB_T, self).__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = nn.InstanceNorm2d(dim)
        
        self.pwconv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.act = nn.GELU()
        
        self.pwconv2 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x+shortcut) 
        x = self.act(x)
        x = self.pwconv2(x)
        return x + shortcut
    
class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1
        self.padding = (kernel_size - 1)//2  
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)  
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1   
        self.weight = nn.Parameter(weight_tensor)  
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(x+y)    

class CGMRNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, only_residual=False):   
        super(CGMRNet, self).__init__()
        self.only_residual = only_residual    

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        self.deconv1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.norm1 = nn.InstanceNorm2d(128, affine=True)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   


        self.deconv21 = nn.Conv2d(64, 128, 3, 2, 1)
        self.norm21 = nn.InstanceNorm2d(128, affine=True)
        self.deconv22 = nn.Conv2d(128, 256, 3, 2, 1)
        self.norm22 = nn.InstanceNorm2d(256, affine=True)      
        self.upconv21 = nn.ConvTranspose2d(256, 128, 4, 2, 1)   
        self.upconv22 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   

        self.res1 = LFEB_F(128)  
        self.res2 = LFEB_T(256)  
        self.res3 = SmoothDilatedResidualBlock(64, dilation=2)

        self.gate = CGAF(64)

        self.postprocess = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1,groups=64, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_c, 1, 1, 0, bias=False),
        )   


    def forward(self, x):

        originimage = x

        x = self.preprocess(x)
        x1 = x
        x2 = x 
        x3 = x

        x1 = self.deconv1(x1)
        x1 = F.relu(self.norm1(x1))
        x1 = self.res1(x1)
        x1 = self.res1(x1)
        y1 = self.upconv1(x1)

        x2 = self.res3(x2)        
        y2 = self.res3(x2)     


        x3 = self.deconv21(x3)
        x3 = F.relu(self.norm21(x3))  
        x3 = self.deconv22(x3)
        x3 = F.relu(self.norm22(x3))
        x3 = self.res2(x3)
        x3 = self.res2(x3)
        x3 = self.upconv21(x3)
        y3 = self.upconv22(x3)

        gated_y = self.gate(y1, y2, y3)  

        y= self.postprocess(gated_y)
        
        return originimage + y 