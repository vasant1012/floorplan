"""-------------------------------------------------------------------------------------"""        
"""                     Complete implementation of UNET                                 """
"""-------------------------------------------------------------------------------------""" 

import torch
import torch.nn as nn
import torch.nn.functional as F

"""-------------------------------------------------------------------------------------"""
"""                                 double_conv                                       """
"""-------------------------------------------------------------------------------------"""
class double_conv(nn.Module):
    #(conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch,dilation):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation,dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

"""-------------------------------------------------------------------------------------"""
"""                                 initial_conv                                       """
"""-------------------------------------------------------------------------------------"""        
class initial_conv(nn.Module):
    #(conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(initial_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1,3), padding=(0,1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_ch, out_ch, (1,3), padding=(0,1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
         
        )

    def forward(self, x):
        x = self.conv(x)
        return x


"""-------------------------------------------------------------------------------------"""
"""                                 inconv                                       """
"""-------------------------------------------------------------------------------------"""
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch,dilation=2)

    def forward(self, x):
        x = self.conv(x)
        return x

"""-------------------------------------------------------------------------------------"""
"""                                 down                                       """
"""-------------------------------------------------------------------------------------"""
class down(nn.Module):
    def __init__(self, in_ch, out_ch,dilation):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch,dilation=dilation)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


"""-------------------------------------------------------------------------------------"""
"""                                 up & forward                                       """
"""-------------------------------------------------------------------------------------"""
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch,dilation=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

"""-------------------------------------------------------------------------------------"""
"""                                 outconv                                       """
"""-------------------------------------------------------------------------------------"""
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


     
class feature_extraction_model_withdropout(nn.Module): #bce
    def __init__(self, n_channels=3, output_ch=10,start_n=8):
        super(feature_extraction_model_withdropout, self).__init__()
        start_n = start_n
        self.dropout = nn.Dropout2d(0.0)
        self.inc   = inconv(n_channels, start_n)
        self.down1 = down(start_n, start_n*2,1)
        self.down2 = down(start_n*2, start_n*4,2)
        self.down3 = down(start_n*4, start_n*8,3)
        self.down4 = down(start_n*8, start_n*8,4)
        self.up1 = up(start_n*16, start_n*8)
        self.up2 = up((start_n*4+start_n*8), start_n*4)
        self.up3 = up((start_n*4+start_n*2), start_n*2)
        self.up4 = up((start_n*2+start_n), start_n)
        self.outc = outconv(start_n, output_ch)
	

    def forward(self, x):# [8,3,512,512]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.dropout(self.down3(x3))
        x5 = self.dropout(self.down4(x4))
        x = self.dropout(self.up1(x5, x4))
        x = self.dropout(self.up2(x, x3))
        x = self.dropout(self.up3(x, x2))
        x = self.up4(x, x1)
        x = self.outc(x)
        return x       

class resize(nn.Module):# Half scale
    def __init__(self) -> None:
        super().__init__()
        self.max_pool=nn.MaxPool2d(2,2)
    def forward(self,x):
        return self.max_pool(x)

# import numpy as np
# img=np.random.random((1,3,1024,1024))
# img=torch.FloatTensor(img)
# obj=resize()
# resized_img=obj(img)
# print('resied_shape : ',resized_img.shape)


