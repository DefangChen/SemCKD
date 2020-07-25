python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256 --epochs 1 --dataset imagenet \
--print-freq 100 --num_workers 8 --distill kd --model_s ResNet18 -r 1 -a 1 -b 0 --trial release \
--learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4
