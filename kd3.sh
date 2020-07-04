python3 train_teacher.py --model ResNet18 --batch_size 64 --epochs 90 --learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4 --dataset imagenet --gpu_id 0,1,2,3
