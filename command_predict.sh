python predict.py \
--input /ssd1/DynamicConv/Detectron_Datasets/cityscapes/leftImg8bit/val/frankfurt  \
--dataset cityscapes \
--model deeplabv3plus_resnet101 \
--ckpt pretrained/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar \
--save_val_results_to test_results
