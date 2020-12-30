# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

# kd
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 1 -a 1 -b 0 --trial 1
# FitNet
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet8x4 -r 1 -a 1 -b 100 --trial 1
# AT
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet8x4 -r 1 -a 1 -b 1000 --trial 1
# SP
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet8x4 -r 1 -a 1 -b 3000 --trial 1
# CC
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet8x4 -r 1 -a 1 -b 0.02 --trial 1
# VID
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet8x4 -r 1 -a 1 -b 1 --trial 1
# RKD
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8x4 -r 1 -a 1 -b 1 --trial 1
# PKT
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8x4 -r 1 -a 1 -b 30000 --trial 1
# CRD
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -r 1 -a 1 -b 0.8 --trial 1
# IRG
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill irg --model_s resnet8x4 -r 1 -a 1 -b 0.005 --trial 1
# SemCKD
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill semckd --model_s resnet8x4 -r 1 -a 1 -b 400 --trial 1
# HKD(two pass)
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8x4_double -r 1 -a 1 -b 30000 --trial 1
python train_student.py --path-t ./save/student_model/S:resnet8x4_double_T:resnet32x4_cifar100_pkt_r:1.0_a:1.0_b:30000.0_1/resnet8x4_double_best.pth  \
--distill hkd --model_s resnet8x4 -r 1 -a 1 -b 1 --trial 1