#python clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'bcva' --batch_size 64
## work
#python clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'patient' --batch_size 96 --epochs 25

python clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'SimCLR' --batch_size 96 --epochs 25
python clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'bcva' --batch_size 96 --epochs 25
python clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'cst' --batch_size 96 --epochs 25
python clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'eye_id' --batch_size 96 --epochs 25


python clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'patient' \
--batch_size 64 --epochs 25 --model 'vit_small_patch16_224' --optim 'AdamW' --learning_rate 2e-5


python clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'bcva' \
--batch_size 96 --epochs 25 --model 'vit_small_patch16_224' --optim 'AdamW' --learning_rate 2e-5


# linear
#python main_linear.py \
#--dataset 'Prime'  \
#--multi 0  \
#--super 4  \
#--results_dir "/home/gu721/yzc/OLIVES_Biomarker/Finetune_OCT_Clinical/unlock-layer4-fluid_srf" \
#--biomarker 'fluid_srf' \
#--ckpt "/home/gu721/yzc/OLIVES_Biomarker/save/SupCon/Prime_TREX_DME_Fixed_models/patient_n_n_n_n_1_1_10_Prime_TREX_DME_Fixed_lr_resnet18_0.001_decay_0.0001_bsz_64_temp_0.07_trial_0__0/last.pth"

