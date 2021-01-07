"""
the general training framework
"""

from __future__ import print_function

import os
import re
import argparse
import time

import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger

# import apex

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed, SelfA

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader, imagenet_list
from dataset.imagenet_dali import get_dali_data_loader

from helper.util import adjust_learning_rate, save_dict_to_json, reduce_tensor

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss, SemCKDLoss
from distiller_zoo import IRGLoss, HKDLoss, PKT
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate

split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    # basic
    parser.add_argument('--print-freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'imagenette'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'ResNet18', 'ResNet34', 'resnet8x4_double',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_50_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg11_imagenet', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_Imagenet', 'MobileNetV2_Imagenet',
                                 'shufflenet_v2_x0_5', 'shufflenet_v2_x2_0', 'ResNet18Double'])
    parser.add_argument('--path-t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', 'vid',
                                                                      'correlation', 'rkd', 'pkt', 'crd', 'semckd',
                                                                      'irg', 'hkd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])

    # transform layers for IRG
    parser.add_argument('--transform_layer_t', nargs='+', type=int, default = [])
    parser.add_argument('--transform_layer_s', nargs='+', type=int, default = [])

    # switch for edge transformation
    parser.add_argument('--no_edge_transform', action='store_true') # default=false
    
    parser.add_argument('--use-lmdb', action='store_true') # default=false

    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)

    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')

    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    parser.add_argument('--hkd_initial_weight', default=100, type=float, help='Initial layer weight for HKD method')
    parser.add_argument('--hkd_decay', default=0.7, type=float, help='Layer weight decay for HKD method')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_r', '{}_a', '{}_b', '{}_{}'])
    opt.model_name = model_name_template.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    if segments[0] == 'resnext50':
        return segments[0] + '_' + segments[1]
    if segments[0] == 'vgg13' and segments[1] == 'imagenet':
        return segments[0] + '_' + segments[1]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    # TODO: reduce size of the teacher saved in train_teacher.py
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model

total_time = time.time()
best_acc = 0

def main():
    
    opt = parse_option()
    
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)

    class_num_map = {
        'cifar100': 100,
        'imagenet': 1000,
        'imagenette': 10,
    }
    if opt.dataset not in class_num_map:
        raise NotImplementedError(opt.dataset)
    n_cls = class_num_map[opt.dataset]

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    module_args = {'num_classes': n_cls}
    model_s = model_dict[opt.model_s](**module_args)
    
    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'semckd':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
       
        criterion_kd = SemCKDLoss()
        self_attention = SelfA(len(feat_s)-2, len(feat_t)-2, opt.batch_size, s_n, t_n)
        module_list.append(self_attention)
        trainable_list.append(self_attention)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = 50000
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'irg':
        criterion_kd = IRGLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'hkd':
        criterion_kd = HKDLoss(init_weight=opt.hkd_initial_weight, decay=opt.hkd_decay)
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    module_list.append(model_t)
    
    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
    elif opt.dataset in imagenet_list:
        if opt.dali is None:
            train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        multiprocessing_distributed=opt.multiprocessing_distributed)
        else:
            train_loader, val_loader = get_dali_data_loader(opt)
    else:
        raise NotImplementedError(opt.dataset)

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if not opt.skip_validation:
        # validate teacher accuracy
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)

        if opt.dali is not None:
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')
    
    # routine
    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss, data_time = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss, data_time]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss, data_time = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}, Data {:.2f}'.format(epoch, opt.gpu, train_acc, train_acc_top5, time2 - time1, data_time))
            
            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        print('GPU %d validating' % (opt.gpu))
        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)        

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
            
            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            logger.log_value('test_acc_top5', test_acc_top5, epoch)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }
                if opt.distill == 'semckd':
                    state['attention'] = trainable_list[-1].state_dict()
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                
                test_merics = {'test_loss': test_loss,
                                'test_acc': test_acc,
                                'test_acc_top5': test_acc_top5,
                                'epoch': epoch}
                
                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)
                
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)
        
        # save parameters
        save_state = {k: v for k, v in opt._get_kwargs()}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model_s.parameters())/1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] =  (time.time() - total_time)/3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json") 
        save_dict_to_json(save_state, params_json_path)

if __name__ == '__main__':
    main()
