# SemCKD

This repo is forked from https://github.com/HobbitLong/RepDistiller. Thanks to its framework, we can add SemCKD and some other methods easily.

To get most pretrained teacher models for CIFAR-100:

```
sh scripts/fetch_pretrained_teachers.sh
```

For ImageNet, pretrained models from torchvision are used, e.g. [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth). Save the model to ./save/models/$MODEL_vanilla/ and use scripts/model_transform.py to make it readable by our code.

Running SemCKD:

```bash
# CIFAR-100
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill semckd --model_s resnet8x4 -r 1 -a 1 -b 400 --trial 0
# ImageNet
python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256 --epochs 90 --dataset imagenet --gpu_id 0,1,2,3,4,5,6,7 --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers 32 --distill semckd --model_s ResNet18 -r 1 -a 1 -b 50 --trial 0 \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali gpu
```

## Citation

## Acknowledgement
谢谢 RepDistiller 仓库的作者