import torch.nn as nn
import torch


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
            
        self.conv1 = CMBBolock(in_channels=3, out_channels=16, kernel_size=3, padding='same')
        self.conv2 = CMBBolock(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.conv3 = CMBBolock(in_channels=32, out_channels=64, kernel_size=3, padding='same')

        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        # self.conv4 = CMBBolock(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        # self.conv5 = CMBBolock(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        # self.conv6 = CMBBolock(in_channels=256, out_channels=256, kernel_size=3, padding='same')

    def forward(self, x):
        conv_block_1 = self.conv1(x)
        conv_block_2 = self.conv2(conv_block_1)
        conv_block_3 = self.conv3(conv_block_2)

        conv_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', bias=False)(conv_block_3)
        conv_4 = self.relu(conv_4)
        maxpool2d_4 = self.maxpool2d(conv_4) # 20X20X128
        bnorm_4 = nn.BatchNorm2d(num_features=128)(maxpool2d_4)

        conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', bias=False)(bnorm_4)
        conv_5 = self.relu(conv_5)
        maxpool2d_5 = self.maxpool2d(conv_5) # 10x10x256
        bnorm_5 = nn.BatchNorm2d(num_features=256)(maxpool2d_5)

        conv_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', bias=False)(bnorm_5)
        conv_6 = self.relu(conv_6)
        maxpool2d_6 = self.maxpool2d(conv_6) # 5x5x256

        class_20x20 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same')(maxpool2d_4)
        class_20x20_reshape = Reshape(last_dim_shape=16)(class_20x20)

        box_20x20 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=3, padding='same')(maxpool2d_4)
        box_20x20_reshape = Reshape(last_dim_shape=4)(box_20x20)

        class_10x10 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding='same')(maxpool2d_5)
        class_10x10_reshape = Reshape(last_dim_shape=16)(class_10x10)

        box_10x10 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding='same')(maxpool2d_5)
        box_10x10_reshape = Reshape(last_dim_shape=4)(box_10x10)

        class_5x5 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding='same')(maxpool2d_6)
        class_5x5_reshape = Reshape(last_dim_shape=16)(class_5x5)

        box_5x5 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding='same')(maxpool2d_6)
        box_5x5_reshape = Reshape(last_dim_shape=4)(box_5x5)

        # print(class_20x20.shape, class_10x10.shape, class_5x5.shape)
        # print(class_20x20_reshape.shape, class_10x10_reshape.shape, class_5x5_reshape.shape)

        class_out = Concatenate(dim=1)([class_20x20_reshape, class_10x10_reshape, class_5x5_reshape])
        box_out = Concatenate(dim=1)([box_20x20_reshape, box_10x10_reshape, box_5x5_reshape])

        final_output = Concatenate(dim=2)([box_out, class_out])
        return final_output
    

    def infer(self, input):
        input = torch.Tensor(input)
        return self.forward(torch.permute(input, (0, 3, 1, 2)))







class CMBBolock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CMBBolock, self).__init__()
        self.cmb = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                # stride=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.cmb(x)







class Reshape(nn.Module):
    def __init__(self, last_dim_shape):
        super(Reshape, self).__init__()
        self.ld_shape = last_dim_shape

    def forward(self, x):
        return torch.reshape(x, (x.shape[0], -1, self.ld_shape))
    






class Concatenate(nn.Module):
    def __init__(self, dim=1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)
    


if __name__ == '__main__':
    rand_sample = torch.rand(24, 3, 320, 320)
    model = Detector()
    print(model(rand_sample).shape)