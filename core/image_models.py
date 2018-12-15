import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
# applied extract visual attributes(concepts)
num_concepts = 111156
class resnet_mil(nn.Module):
    def __init__(self):
        super(resnet_mil, self).__init__()

        resnet = models.resnet101(pretrained=True)
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1", resnet.conv1)
        self.conv.add_module("bn1", resnet.bn1)
        self.conv.add_module("relu", resnet.relu)
        self.conv.add_module("maxpool", resnet.maxpool)
        self.conv.add_module("layer1", resnet.layer1)
        self.conv.add_module("layer2", resnet.layer2)
        self.conv.add_module("layer3", resnet.layer3)
        self.conv.add_module("layer4", resnet.layer4)
        self.l1 = nn.Sequential(nn.Linear(2048, 92),
                                nn.ReLU(True)
                                )
        self.att_size = 7
        self.pool_mil = nn.MaxPool2d(kernel_size=self.att_size, stride=0)
        self._initialize_weights()

    def forward(self, img, att_size=14):
        x0 = self.conv(img)
        x = self.pool_mil(x0)
        x = x.squeeze(2).squeeze(2)

        x = self.l1(x)
        x1 = torch.add(torch.mul(x.view(x.size(0), 92, -1), -1), 1)
        cumprod = torch.cumprod(x1, 2)
        out = torch.max(x, torch.add(torch.mul(cumprod[:, :, -1], -1), 1))
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    init.xavier_normal(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class vgg16_mil(nn.Module):
    def __init__(self, concepts_size=num_concepts):
        super(vgg16_mil, self).__init__()
        self.concepts_size = concepts_size
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self._vgg_extractor = nn.Sequential(*(vgg16_bn.features[i] for i in xrange(43)))
        for p in self._vgg_extractor.parameters():
            p.requires_grad = False

        # fine-tune the logit layer.

        self.logit = nn.Sequential()
        self.logit.add_module("fc6_conv", nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0))
        self.logit.add_module("relu_6", nn.ReLU())
        self.logit.add_module("fc7_conv", nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0))
        self.logit.add_module("relu_7", nn.ReLU())
        self.logit.add_module("fc8_conv", nn.Conv2d(4096, concepts_size, kernel_size=1, stride=1, padding=0))
        self.logit.add_module("sigmoid_8", nn.Sigmoid())

        self.pool_mil = nn.MaxPool2d(kernel_size=29, stride=0)

    def forward(self, img):
        x0 = self._vgg_extractor(img)
        # possibility of each concept given image region j in image i corresponds to word concept
        odds = self.logit.forward(x0)
        # print odds.shape
        x = self.pool_mil(odds)

        x = x.squeeze(2).squeeze(2)

        # equation (1) in paper <<From Captions to Visual Concepts and Back>>
        x1 = torch.add(torch.mul(odds.view(x.size(0), self.concepts_size, -1), -1), 1)
        cumprod = torch.cumprod(x1, 2)

        out = torch.max(x, torch.add(torch.mul(cumprod[:, :, -1], -1), 1))
        # print out
        return out


class VGG(nn.Module):

    def __init__(self, num_classes=num_concepts):
        super(VGG, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self._vgg_extractor = nn.Sequential(*(vgg16_bn.features[i] for i in xrange(43)))
        # for p in self._vgg_extractor.parameters():
        #     p.requires_grad = False
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.features = self.resnet
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2 * 2, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    def generate_vis(self, output, topk = 23):
        values, indices = torch.topk(output, topk)
        return indices
    def predict(self, output, topk= 10):
        """
        :param output: the probability of each label.
        :param topk: accuracy of top k,
        :return: binary output
        """
        threshold = 1.0 / num_concepts
        # print output.shape
        output = F.softmax(output, 1)
        output[output >= threshold] = 1
        output[output < threshold] = 0
        return output

# class vgg_mil(nn.Module):
#     def __init__(self):
#         super(vgg_mil, self).__init__()
#         self.conv = torch.nn.Sequential()
#         self.conv.add_module("conv1_1", nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_1_1", torch.nn.ReLU())
#         self.conv.add_module("conv1_2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_1_2", torch.nn.ReLU())
#         self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
#
#         self.conv.add_module("conv2_1", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_2_1", torch.nn.ReLU())
#         self.conv.add_module("conv2_2", nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_2_2", torch.nn.ReLU())
#         self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
#
#         self.conv.add_module("conv3_1", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_3_1", torch.nn.ReLU())
#         self.conv.add_module("conv3_2", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_3_2", torch.nn.ReLU())
#         self.conv.add_module("conv3_3", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_3_3", torch.nn.ReLU())
#         self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=2))
#
#         self.conv.add_module("conv4_1", nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_4_1", torch.nn.ReLU())
#         self.conv.add_module("conv4_2", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_4_2", torch.nn.ReLU())
#         self.conv.add_module("conv4_3", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_4_3", torch.nn.ReLU())
#         self.conv.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=2))
#
#         self.conv.add_module("conv5_1", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_5_1", torch.nn.ReLU())
#         self.conv.add_module("conv5_2", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_5_2", torch.nn.ReLU())
#         self.conv.add_module("conv5_3", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
#         self.conv.add_module("relu_5_3", torch.nn.ReLU())
#         self.conv.add_module("maxpool_5", torch.nn.MaxPool2d(kernel_size=2))
#
#         self.conv.add_module("fc6_conv", nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0))
#         self.conv.add_module("relu_6_1", torch.nn.ReLU())
#
#         self.conv.add_module("fc7_conv", nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0))
#         self.conv.add_module("relu_7_1", torch.nn.ReLU())
#
#         self.conv.add_module("fc8_conv", nn.Conv2d(4096, num_concepts, kernel_size=1, stride=1, padding=0))
#         self.conv.add_module("sigmoid_8", torch.nn.Sigmoid())
#
#         self.pool_mil = nn.MaxPool2d(kernel_size=11, stride=0)
#
#         # self.weight_init()
#
#     # def weight_init(self):
#     #     self.cnn_weight = 'model/vgg16_full_conv_mil.pth'
#     #     self.conv.load_state_dict(torch.load(self.cnn_weight))
#     #     print("Load pretrained CNN model from " + self.cnn_weight)
#
#     def forward(self, x):
#         x0 = self.conv.forward(x.float())
#         x = self.pool_mil(x0)
#         x = x.squeeze(2).squeeze(2)
#         x1 = torch.add(torch.mul(x0.view(x.size(0), num_concepts, -1), -1), 1)
#         cumprod = torch.cumprod(x1, 2)
#         out = torch.max(x, torch.add(torch.mul(cumprod[:, :, -1], -1), 1))
#         out = F.softmax(out)
#         return out
# applied to extract image features.
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        # this layer is modifiable
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

if __name__ == '__main__':
    model = EncoderCNN(256)
    # 565, 565   ---> 29, 29
    sample = torch.rand(1, 3, 224, 224)
    print model.forward(sample)
    # for name, param in model.named_parameters():
    #     print name, param.size()