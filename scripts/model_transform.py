import torch

if __name__ == '__main__':
    state_dict = torch.load('save/models/vgg13_imagenet_vanilla/vgg13_bn-abd245e5.pth')
    torch.save({
        # 'epoch': model['epoch'],
        'model': state_dict, 
        # 'best_acc': model['best_acc1']
    }, 'save/models/vgg13_imagenet_vanilla/vgg13_transformed.pth')