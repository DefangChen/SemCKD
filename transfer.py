from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import tensorboard_logger as tb_logger
from os import path
import time
import copy
import argparse
import json
import re

from models import model_dict


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--model', type=str, default='vgg8', choices=['vgg8', 'resnet32x4', 'resnet8x4', 'vgg13',
                                                                  'ShuffleV2', 'MobileNetV2', 'wrn_40_2'])
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--target-dataset', type=str, default='stl10', choices=['stl10', 'tiny-imagenet'])
parser.add_argument('--id', type=str, default='0')
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--epoch', type=int, default=240)

# optimization
parser.add_argument('--learning-rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--lr-decay-epochs', nargs='+', default=[150, 180, 210], help='where to decay lr')
parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

args = parser.parse_args()

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'test': transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
}

if args.target_dataset == 'stl10':
    data_dir = 'data/stl10'
    datasets = {x: datasets.STL10(data_dir, split=x, transform=data_transforms[x]) for x in ['train', 'test']}
    num_classes = 10
elif args.target_dataset == 'tiny-imagenet':
    data_dir = 'data/tiny-imagenet-200'
    datasets = {
        'train': datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train']),
        'test': datasets.ImageFolder(data_dir + '/val', transform=data_transforms['test'])
    }
    num_classes = 200

dataloader = {x: torch.utils.data.DataLoader(datasets[x], batch_size=128,
                                             shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}

device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

directory = args.model_path.split('/')[-2]
pattern = '.*cifar100_(.+)_r'
method = re.match(pattern, directory)
if method is not None:
    method = method[1]
else:
    method = 'none'
tb_folder = path.join('save', 'tensorboard', '_'.join(['transfer', args.model, args.target_dataset, args.id, method]))
logger = tb_logger.Logger(logdir=tb_folder, flush_secs=2)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    results = [0 for _ in range(num_epochs)]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}({:d}/{:d})'.format(
                phase, epoch_loss, epoch_acc, running_corrects, dataset_sizes[phase]))

            results[epoch] = epoch_acc

            logger.log_value('%s_acc' % phase, epoch_acc, epoch)
            logger.log_value('%s_loss' % phase, epoch_loss, epoch)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, best_epoch, results


if __name__ == '__main__':
    # Load pretrained model
    model_conv = model_dict[args.model](num_classes=100)
    model_state_dict = torch.load(args.model_path)['model']
    model_conv.load_state_dict(model_state_dict)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    if 'vgg' in args.model:
        num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = classifier = nn.Linear(num_ftrs, num_classes)
    elif 'resnet' in args.model or 'wrn' in args.model:
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = classifier = nn.Linear(num_ftrs, num_classes)
    elif 'Shuffle' in args.model:
        num_ftrs = model_conv.linear.in_features
        model_conv.linear = classifier = nn.Linear(num_ftrs, num_classes)
    elif 'MobileNetV2' in args.model:
        num_ftrs = model_conv.classifier[0].in_features
        model_conv.classifier = classifier = nn.Linear(num_ftrs, num_classes)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(classifier.parameters(), lr=args.learning_rate,
                               momentum=0.9, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_conv, milestones=args.lr_decay_epochs,
                                                gamma=args.lr_decay_rate)

    model_conv, acc, epoch, acc_list = train_model(model_conv, criterion, optimizer_conv,
                                                   exp_lr_scheduler, num_epochs=args.epoch)

    # save model
    directory = path.dirname(args.model_path)
    filename = path.join(directory, 'transfer_%s_%s.pth' % (args.id, args.target_dataset))
    torch.save(model_conv.state_dict(), filename)

    # save acc
    jsonfile = path.join(directory, 'transfer_%s_%s.json' % (args.id, args.target_dataset))
    with open(jsonfile, 'w+') as f:
        json.dump({'best_acc': acc.item(), 'epoch': epoch}, f)

    torch.save(acc_list, path.join('save', 'transfer', '%s_%s_%s_%s_acclist.pth' % (args.id, args.target_dataset,
                                                                                    args.model, method)))
