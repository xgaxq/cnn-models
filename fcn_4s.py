import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class VGG_19bn_4s(nn.Module):
    def __init__(self, n_class=21):
        super(VGG_19bn_4s, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = nn.Conv2d(512, 4096, 7)  # padding=0
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.trans_f4 = nn.Conv2d(512, n_class, 1)
        self.trans_f3 = nn.Conv2d(256, n_class, 1)
        self.trans_f2 = nn.Conv2d(128, n_class, 1)

        self.up2times = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.up4times = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.up8times = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.up32times = nn.ConvTranspose2d(
            n_class, n_class, 8, stride=4, bias=False)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = bilinear_kernel(n_class, n_class, m.kernel_size[0])

    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))

        f6 = self.drop6(self.relu6(self.fc6(f5)))
        f7 = self.score_fr(self.drop7(self.relu7(self.fc7(f6))))

        up2_feat = self.up2times(f7)
        h = self.trans_f4(f4)
        print(h.shape)
        print(up2_feat.shape)
        h = h[:, :, 5:5 + up2_feat.size(2), 5:5 + up2_feat.size(3)]
        h = h + up2_feat

        up4_feat = self.up4times(h)
        h = self.trans_f3(f3)
        print(h.shape)
        print(up4_feat.shape)
        h = h[:, :, 9:9 + up4_feat.size(2), 9:9 + up4_feat.size(3)]
        h = h + up4_feat

        up8_feat = self.up8times(h)
        h = self.trans_f2(f2)
        print(h.shape)
        print(up8_feat.shape)
        h = h[:, :, 17:17 + up8_feat.size(2), 17:17 + up8_feat.size(3)]
        h = h + up8_feat

        h = self.up32times(h)
        print(h.shape)
        final_scores = h[:, :, 33:33 + x.size(2), 33:33 + x.size(3)].contiguous()

        return final_scores

if __name__ == '__main__':
    model = VGG_19bn_4s()
    print(model)