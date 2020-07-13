python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256  --epochs 90 --dataset imagenet --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23334 \
--print-freq 100 --num_workers 16 --distill hint  --model_s ResNet18 -r 1 -a 1 -b 100 --trial 0 \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --hint_layer 1 \
--dali gpu
