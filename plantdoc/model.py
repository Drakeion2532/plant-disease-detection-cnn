"""CNN from Chohan et al. (2020), 'Plant Disease Detection using Deep Learning':
five 3x3 convolution blocks, each followed by batch normalisation and 2x2
max-pooling, then a dense layer and a 38-way softmax."""
import torch
from torch import nn


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class PlantCNN(nn.Module):
    def __init__(self, n_classes=38, widths=(32, 64, 128, 256, 256), img_size=96, dropout=0.4):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers.append(conv_block(c, w))
            c = w
        self.features = nn.Sequential(*layers)
        s = img_size // 2 ** len(widths)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c * s * s, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    m = PlantCNN()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
    print(m(torch.zeros(2, 3, 96, 96)).shape)
