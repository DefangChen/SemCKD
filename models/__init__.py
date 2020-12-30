from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnet import resnet8x4_double
from .resnetv2 import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d, resnet34x4, resnet18x2
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .vggv2 import vgg13_bn as vgg13_imagenet, vgg11_bn as vgg11_imagenet
from .mobilenetv2 import mobile_half
from .mobilenetv2_imagenet import mobilenet_v2
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .ShuffleNetv2_Imagenet import shufflenet_v2_x1_0 as ShuffleNetV2Imagenet, shufflenet_v2_x0_5, shufflenet_v2_x2_0

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'ResNet18': resnet18,
    'ResNet18Double': resnet18x2,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet8x4_double': resnet8x4_double,
    'resnet32x4': resnet32x4,
    'resnext50_32x4d': resnext50_32x4d,
    'resnet34x4': resnet34x4,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_50_2': wide_resnet50_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'vgg13_imagenet': vgg13_imagenet,
    'vgg11_imagenet': vgg11_imagenet,
    'MobileNetV2': mobile_half,
    'MobileNetV2_Imagenet': mobilenet_v2,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_Imagenet': ShuffleNetV2Imagenet,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
}
