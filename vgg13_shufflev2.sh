BATCH_SIZE=256
WORKER=32
GPU=0,1,2,3,4,5,6,7
EPOCH=90
LEARNING_RATE=0.1
DALI=cpu

# aakd
python train_student.py --path-t ./save/models/vgg13_imagenet_vanilla/vgg13_transformed.pth \
--batch_size $BATCH_SIZE --epochs $EPOCH --dataset imagenet --gpu_id $GPU --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers $WORKER --model_s ShuffleV2_Imagenet  --trial release \
--multiprocessing-distributed --learning_rate $LEARNING_RATE --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali $DALI \
--distill aakd -r 1 -a 1 -b 400
# kd
python train_student.py --path-t ./save/models/vgg13_imagenet_vanilla/vgg13_transformed.pth \
--batch_size $BATCH_SIZE --epochs $EPOCH --dataset imagenet --gpu_id $GPU --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers $WORKER --model_s ShuffleV2_Imagenet  --trial release \
--multiprocessing-distributed --learning_rate $LEARNING_RATE --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali $DALI \
--distill kd -r 1 -a 1 -b 0
# fitnet
python train_student.py --path-t ./save/models/vgg13_imagenet_vanilla/vgg13_transformed.pth \
--batch_size $BATCH_SIZE --epochs $EPOCH --dataset imagenet --gpu_id $GPU --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers $WORKER --model_s ShuffleV2_Imagenet  --trial release \
--multiprocessing-distributed --learning_rate $LEARNING_RATE --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali $DALI \
--distill hint -r 1 -a 1 -b 100 --hint-layer 1
# sp
python train_student.py --path-t ./save/models/vgg13_imagenet_vanilla/vgg13_transformed.pth \
--batch_size $BATCH_SIZE --epochs $EPOCH --dataset imagenet --gpu_id $GPU --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers $WORKER --model_s ShuffleV2_Imagenet  --trial release \
--multiprocessing-distributed --learning_rate $LEARNING_RATE --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali $DALI \
--distill sp -r 1 -a 1 -b 3000
# vid
python train_student.py --path-t ./save/models/vgg13_imagenet_vanilla/vgg13_transformed.pth \
--batch_size $BATCH_SIZE --epochs $EPOCH --dataset imagenet --gpu_id $GPU --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers $WORKER --model_s ShuffleV2_Imagenet  --trial release \
--multiprocessing-distributed --learning_rate $LEARNING_RATE --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali $DALI \
--distill vid -r 1 -a 1 -b 1
# irg
python train_student.py --path-t ./save/models/vgg13_imagenet_vanilla/vgg13_transformed.pth \
--batch_size $BATCH_SIZE --epochs $EPOCH --dataset imagenet --gpu_id $GPU --dist-url tcp://127.0.0.1:23333 \
--print-freq 100 --num_workers $WORKER --model_s ShuffleV2_Imagenet  --trial release \
--multiprocessing-distributed --learning_rate $LEARNING_RATE --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali $DALI \
--distill irg -r 1 -a 1 -b 0.05
