
import torch
import torch.nn as nn
from base_model import conv_block_nested
from base_model import output_block_conv
class NestUnet(nn.Module):

    def __init__(self, in_ch=6, out_ch=1):
        super(NestUnet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])


        self.final = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1),nn.Sigmoid())
        self.output = output_block_conv(filters[0],1)

    def forward(self, x, y):


        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))





        y0_0 = self.conv0_0(y)
        y1_0 = self.conv1_0(self.pool(y0_0))
        y0_1 = self.conv0_1(torch.cat([y0_0, self.Up(y1_0)], 1))

        y2_0 = self.conv2_0(self.pool(y1_0))
        y1_1 = self.conv1_1(torch.cat([y1_0, self.Up(y2_0)], 1))
        y0_2 = self.conv0_2(torch.cat([y0_0, y0_1, self.Up(y1_1)], 1))

        y3_0 = self.conv3_0(self.pool(y2_0))
        y2_1 = self.conv2_1(torch.cat([y2_0, self.Up(y3_0)], 1))


        y4_0 = self.conv4_0(self.pool(y3_0))




        #相互融合
        xy0_0 = torch.abs(x0_0-y0_0)
        xy1_0 = torch.abs(x1_0-y1_0)
        xy2_0 = torch.abs(x2_0-y2_0)
        xy3_0 = torch.abs(x3_0-y3_0)

        xy0_1 = torch.abs(x0_1-y0_1)
        xy0_2 = torch.abs(x0_2-y0_2)
        xy1_1 = torch.abs(x1_1-y1_1)
        xy2_1 = torch.abs(x2_1-y2_1)
        xy4_0 = torch.abs(x4_0-y4_0)

        xy1_2 = self.conv1_2(torch.cat([xy1_0, xy1_1, self.Up(xy2_1)], 1))
        xy3_1 = self.conv3_1(torch.cat([xy3_0, self.Up(xy4_0)], 1))
        xy0_3 = self.conv0_3(torch.cat([xy0_0, xy0_1, xy0_2, self.Up(xy1_2)], 1))
        xy2_2 = self.conv2_2(torch.cat([xy2_0, xy2_1, self.Up(xy3_1)], 1))
        xy1_3 = self.conv1_3(torch.cat([xy1_0, xy1_1, xy1_2, self.Up(xy2_2)], 1))
        xy0_4 = self.conv0_4(torch.cat([xy0_0, xy0_1, xy0_2, xy0_3, self.Up(xy1_3)], 1))
        output1 = self.output(xy0_1)
        output2 = self.output(xy0_2)
        output3 = self.output(xy0_3)
        output4 = self.output(xy0_4)
        output5 = output_block_conv(4,1)(torch.cat([output1,output2,output3,output4],-1))







        output = self.final(xy0_4)
        return output




