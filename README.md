# SemCKD

Cross-Layer Distillation with Semantic Calibration (AAAI-2021) https://arxiv.org/abs/2012.03236

This is a PyTorch-1.0 implementation of SemCKD together with the compared approaches, such as classic KD, Feature-Map Distillation variants like FitNet, AT, SP, VID and HKD.

**Abstract**
Recently proposed knowledge distillation approaches based on feature-map transfer validate that intermediate layers of a teacher model can serve as effective targets for training a student model to obtain better generalization ability. Existing studies mainly focus on particular representation forms for knowledge transfer between manually specified pairs of teacher-student intermediate layers. However, semantics of intermediate layers may vary in different networks and manual association of layers might lead to negative regularization caused by semantic mismatch between certain teacher-student layer pairs. To address this problem, we propose Semantic Calibration for Cross-layer Knowledge Distillation (SemCKD), which automatically assigns proper target layers of the teacher model for each student layer with an attention mechanism. With a learned attention distribution, each student layer distills knowledge contained in multiple layers rather than a single fixed intermediate layer from the teacher model for appropriate cross-layer supervision in training. Consistent improvements over state-of-the-art approaches are observed in extensive experiments with various network architectures for teacher and student models, demonstrating the effectiveness and flexibility of the proposed attention based soft layer association mechanism for cross-layer distillation.

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

**Notes:** 

This repo was forked from https://github.com/HobbitLong/RepDistiller. The main difference in implementation is that we set both weights for classification loss and logit-level distillation loss as **1** throughout the experiments, which is a more common practice for knowledge distillation.  

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

The implementation of compared methods mainly based on the author-provided code and a open-source benchmark https://github.com/HobbitLong/RepDistiller. Thanks to the excellent work of [Yonglong Tian](https://github.com/HobbitLong), we can implement SemCKD and some other methods easily.