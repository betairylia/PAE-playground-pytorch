import torch
from torch import nn

class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).cuda()

    def forward(self, x):
        if self.training:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x

class VMTModel(nn.Module):
    def __init__(self, large = False, bn = True, IN = True, nOut = 10):
        super(VMTModel, self).__init__()

        n_features = 192 if large else 64

        self.feature_extractor = nn.Sequential(
            nn.InstanceNorm2d(3, momentum=1, eps=1e-3),  # L-17
            nn.Conv2d(3, n_features, 3, 1, 1),  # L-16
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-16
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-16
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-15
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-15
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-15
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-14
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-14
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-14
            nn.MaxPool2d(2),  # L-13
            nn.Dropout(0.5),  # L-12
            GaussianNoise(1.0),  # L-11
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-10
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-10
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-10
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-9
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-9
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-9
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-8
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-8
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-8
            nn.MaxPool2d(2),  # L-7
            nn.Dropout(0.5),  # L-6
            GaussianNoise(1.0),  # L-5
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-4
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-4
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-4
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-3
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-3
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-3
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-2
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-2
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-2
            nn.AdaptiveAvgPool2d(1),  # L-1
            nn.Conv2d(n_features, nOut, 1)
        )

        self.n_features = n_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.track_running_stats = False

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def forward(self, x, track_bn=False):

        if track_bn:
            self.track_bn_stats(True)

        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if track_bn:
            self.track_bn_stats(False)

        return features.view(features.size(0), -1), logits.view(x.size(0), 10)

    def output_num(self):
        return self.n_features * 1 * 8 * 8