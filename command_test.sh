python train_affinity.py \
--data_root  '/home/lab/ssd1/DynamicConv/Detectron_Datasets/cityscapes' \
--dataset cityscapes \
--enable_vis \
--vis_port 28333 \
--gpu_id 1 \
--lr 0.01 \
--crop_size 513 \
--batch_size 16 \
--output_stride 16 \
--ckpt /home/lab/ssd1/DynamicConv/Code/DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_resnet50_two_branch_cityscapes_os16.pth \
--test_only \
--save_val_results