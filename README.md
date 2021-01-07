# SemCKD

Cross-Layer Distillation with Semantic Calibration (AAAI-2021) https://arxiv.org/abs/2012.03236

The existing *feature distillation* works can be separated into two categories according to the position where knowledge distillation is performed. One is *feature-map distillation* and another one is *feature-embedding distillation*. SemCKD belongs to *feature-map distillation* and is compatible with SOTA *feature-embedding distillation* (e.g., CRD) to further boost the performance of Student Networks.

This repo contains the PyTorch-1.0 implementation of SemCKD together with the compared approaches, such as classic KD, *Feature-Map Distillation* variants like FitNet, AT, SP, VID, HKD and *feature-embedding distillation* variants like PKT, RKD, IRG, CC, CRD.

To get the pretrained teacher models for CIFAR-100:

```
sh scripts/fetch_pretrained_teachers.sh
```

For ImageNet, pretrained models from torchvision are used, e.g. [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth). Save the model to ./save/models/$MODEL_vanilla/ and use scripts/model_transform.py to make it readable by our code.

Running SemCKD:

```bash
# CIFAR-100
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill semckd --model_s resnet8x4 -r 1 -a 1 -b 400 --trial 0
# ImageNet
python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256 --epochs 90 --dataset imagenet --gpu_id 0,1,2,3,4,5,6,7 --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers 32 --distill semckd --model_s ResNet18 -r 1 -a 1 -b 50 --trial 0 \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali gpu
```

**Notes:** 

The main part of this repo was forked from https://github.com/HobbitLong/RepDistiller. The main difference in implementation is that we set both weights for classification loss and logit-level distillation loss as **1** throughout the experiments, which is a more common practice for knowledge distillation.  

The codes in this repository is merged from different sources, and we have not tested them thoroughly. Hence, if you have any questions, please contact us without hesitation.

## Citation
If you find this repository useful, please consider citing the following paper:
```
@article{chen2020cross,
  title={Cross-Layer Distillation with Semantic Calibration},
  author={Chen, Defang and Mei, Jian-Ping and Zhang, Yuan and Wang, Can and Wang, Zhe and Feng, Yan and Chen, Chun},
  journal={arXiv preprint arXiv:2012.03236},
  year={2020}
}
```

## Acknowledgement

The implementation of compared methods are mainly based on the author-provided code and a open-source benchmark https://github.com/HobbitLong/RepDistiller. Thanks to the excellent work of [Yonglong Tian](https://github.com/HobbitLong), we can implement SemCKD and some other methods easily.
