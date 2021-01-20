import torch.nn as nn
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Generator
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Linear(opt.resSize, 1)
        self.classifier = nn.Linear(opt.resSize, opt.nclass_seen)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x):
        dis_out = self.discriminator(x)
        pred = self.logic(self.classifier(x))
        return dis_out, pred