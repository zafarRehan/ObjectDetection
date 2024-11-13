import torch.nn as nn
import torch
from config import Config, DatasetConfig
from torchsummary import summary


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
            
        self.conv1 = CMBBolock(in_channels=DatasetConfig.IMAGE_CHANNELS, out_channels=16, kernel_size=3, padding='same')
        self.conv2 = CMBBolock(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.conv3 = CMBBolock(in_channels=32, out_channels=64, kernel_size=3, padding='same')

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', bias=True)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', bias=True)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, bias=True)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0, bias=True)


        self.bnorm4 = nn.BatchNorm2d(num_features=128)
        self.bnorm5 = nn.BatchNorm2d(num_features=256)
        self.bnorm6 = nn.BatchNorm2d(num_features=256)


        self.class20x20 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same')     
        self.class10x10 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding='same')
        self.class5x5 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding='same')
        self.class3x3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding='same')
        self.class1x1 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding='same')

        self.box20x20_1 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=3, padding='same')
        self.box10x10_1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding='same')
        self.box5x5_1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding='same')
        self.box3x3_1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding='same')
        self.box1x1_1 = nn.Conv2d(in_channels=512, out_channels=16, kernel_size=3, padding='same')

        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu = nn.ReLU()

        self.out_class_shape = DatasetConfig.N_CLASSES + 1
        self.box_shape = 4

        self.reshape_class = Reshape(last_dim_shape=self.out_class_shape)
        self.reshape_box = Reshape(last_dim_shape=self.box_shape)

        self.checkpoint = None
        

    def forward(self, x):

        conv_block_1 = self.conv1(x)
        conv_block_2 = self.conv2(conv_block_1)
        conv_block_3 = self.conv3(conv_block_2)

        conv_4 = self.conv4(conv_block_3)
        conv_4 = self.relu(conv_4)
        maxpool2d_4 = self.maxpool2d(conv_4) # 20X20X128
        bnorm_4 = self.bnorm4(maxpool2d_4)

        conv_5 = self.conv5(bnorm_4)
        conv_5 = self.relu(conv_5)
        maxpool2d_5 = self.maxpool2d(conv_5) # 10x10x256
        bnorm_5 = self.bnorm5(maxpool2d_5)

        conv_6 = self.conv6(bnorm_5)
        conv_6 = self.relu(conv_6)
        maxpool2d_6 = self.maxpool2d(conv_6) # 5x5x256
        bnorm_6 = self.bnorm6(maxpool2d_6)

        conv_7 = self.conv7(bnorm_6)
        conv_7 = self.relu(conv_7) # 3x3x256

        conv_8 = self.conv8(conv_7)
        conv_8 = self.relu(conv_8) # 1x1x512
 
        class_20x20 = self.class20x20(maxpool2d_4)
        class_20x20_reshape = self.reshape_class(class_20x20)

        box_20x20 = self.box20x20_1(maxpool2d_4)
        box_20x20_reshape = Reshape(last_dim_shape=self.box_shape)(box_20x20)

        class_10x10 = self.class10x10(maxpool2d_5)
        class_10x10_reshape = Reshape(last_dim_shape=self.out_class_shape)(class_10x10)

        box_10x10 = self.box10x10_1(maxpool2d_5)
        box_10x10_reshape = Reshape(last_dim_shape=self.box_shape)(box_10x10)

        class_5x5 = self.class5x5(maxpool2d_6)
        class_5x5_reshape = Reshape(last_dim_shape=self.out_class_shape)(class_5x5)

        box_5x5 = self.box5x5_1(maxpool2d_6)
        box_5x5_reshape = Reshape(last_dim_shape=self.box_shape)(box_5x5)

        class_3x3 = self.class3x3(conv_7)
        class_3x3_reshape = Reshape(last_dim_shape=self.out_class_shape)(class_3x3)

        box_3x3 = self.box3x3_1(conv_7)
        box_3x3_reshape = Reshape(last_dim_shape=self.box_shape)(box_3x3)

        class_1x1 = self.class1x1(conv_8)
        class_1x1_reshape = Reshape(last_dim_shape=self.out_class_shape)(class_1x1)

        box_1x1 = self.box1x1_1(conv_8)
        box_1x1_reshape = Reshape(last_dim_shape=self.box_shape)(box_1x1)

        class_out = Concatenate(dim=1)([class_20x20_reshape, class_10x10_reshape, class_5x5_reshape, class_3x3_reshape, class_1x1_reshape])
        box_out = Concatenate(dim=1)([box_20x20_reshape, box_10x10_reshape, box_5x5_reshape, box_3x3_reshape, box_1x1_reshape])

        final_output = Concatenate(dim=2)([box_out, class_out])
        return final_output
    

    def infer(self, input):
        input = torch.Tensor(input).to(Config.DEVICE)
        return self.forward(torch.permute(input, (0, 3, 1, 2)))
    

    def save(self, optim:torch.optim, step:int, path):
        torch.save({
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'step': step,
                    }, path)


    def load(self, path):
        self.checkpoint = torch.load(path, map_location=torch.device('cpu'))
        print(self.checkpoint.keys())
        self.load_state_dict(self.checkpoint['model_state_dict'])


    def save_torchscript(self, path):
        model_scripted = torch.jit.script(self) # Export to TorchScript
        model_scripted.save(path) # Save


class CMBBolock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CMBBolock, self).__init__()
        self.cmb = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                bias=True,
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
        return torch.reshape(torch.permute(x, (0, 2, 3, 1)), (x.shape[0], -1, self.ld_shape))
        # return torch.reshape(x, (x.shape[0], -1, self.ld_shape))
    






class Concatenate(nn.Module):
    def __init__(self, dim=1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        # return torch.cat(inputs, dim=self.dim).contiguous()
        return torch.cat(inputs, dim=self.dim)
    


if __name__ == '__main__':
    rand_sample = torch.rand(24, 3, 320, 320)
    model = Detector().cuda()
    # print(model(rand_sample).shape)

    for layer in model.state_dict():
        print(layer, )

    print(model)

    summary(model, input_size=(3, 320, 320))

    for parameter in model.parameters(): 
        print(parameter.shape, parameter.requires_grad)