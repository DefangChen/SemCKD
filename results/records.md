<!-- ## kd
```
# kd.sh
python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256 --epochs 90 --dataset imagenet --gpu_id 4,5,6,7 --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers 16 --distill kd --model_s ResNet18 -r 1 -a 1 -b 0 --trial 0 \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali gpu
```
13号晚上 8:38 开始

## fitnet
```
# fitnet.sh
python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256  --epochs 90 --dataset imagenet --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23334 \
--print-freq 100 --num_workers 16 --distill hint  --model_s ResNet18 -r 1 -a 1 -b 100 --trial 0 \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --hint_layer 1 \
--dali gpu
```

13号晚上 8:38 和 kd 同时开始。  
![kd&fitnet截图](kd_fitnet_screenshot.png)
14 号 17:50 分结束  
![kd&fitnet截图](kd_fitnet_finish_screenshot.png)
 -->

<!-- ## kd
```
# kd.sh
python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256 --epochs 90 --dataset imagenet --gpu_id 0,1,2,3,4,5,6,7 --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers 32 --distill kd --model_s ResNet18 -r 1 -a 1 -b 0 --trial test \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali gpu
```
7.15,12:52 开始训练，第一个 epoch 的截图：  
![第一个epoch截图](kd8_epoch1.png)
半夜不知道什么时候结束，得查一查 tensorboard. 结束截图：
![结束](kd8_32x8_32worker_daligpu.png)


## fitnet8
```
# fitnet8.sh
python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256  --epochs 90 --dataset imagenet --gpu_id 0,1,2,3,4,5,6,7 --dist-url tcp://127.0.0.1:23334 \
--print-freq 100 --num_workers 32 --distill hint  --model_s ResNet18 -r 1 -a 1 -b 100 --trial release \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4 --hint_layer 1 \
--dali gpu
```
7.16 下午开始训练。中间暂停了一会。
![GPU截图](fitnet8_GPU.png) -->

上面的实验 validate 都有问题，但是误差极小，不过还是统统注释掉。下面开始时修复的 validate。
## irg8.sh
```
python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256 --epochs 90 --dataset imagenet --gpu_id 0,1,2,3,4,5,6,7 --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers 32 --distill irg --model_s ResNet18 -r 1 -a 1 -b 0.005 --trial release \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali gpu
```
07.17 下午约两点四十开始，GPU 截图:  
![release_irg_GPU](release_irg_GPU.png)

## irg8.sh b=0.05
7.18 早上又跑了 irg8.sh，不过把 b 改成 0.05，GPU 截图
![release_irg_0d05_GPU](release_irg_0d05_GPU.png)
晚上十一点不到结束，截图
![release_irg_0d05](release_irg0dot05.png)

## sp8.sh
7.18 晚上十一点开始跑 sp8.sh, GPU 截图
![release_sp](release_sp8.png)
次日十二点看跑完了，截图
![release_sp_finish](release_sp8_finish.png)

## vid8.sh
7.19 中午跑的，没有截图，半夜跑完。截图：
![release_vid](release_vid.png)  
图里显存占 8g 是因为已经开始跑 aakd 了。

## kd8.sh
7.20 早上 10:43 开始跑。GPU 截图  
![release_kd](release_kd_GPU.png)

## validation
single GPU 73.314