CUDA_VISIBLE_DEVICES="0" \
accelerate launch \
--main_process_port 13647 \
--num_processes 1 \
--gpu_ids 0 \ # Use a Single GPU
../main.py \
--input_path [Your Input Path] \
--save_dir [Your Save Directory] \
--train_type single \
--type_token \
--dpe \
--pos_enc \
--n_layers 2 \
--batch_size 64 \
--wandb_project_name [Your Wandb Project Name] \
--wandb_entity_name [Your Wandb Entity Name] \
--seed 42 \
--src_data mimiciii_cv \ # Client Data to Generate Latent
--mixed_precision no \
--extract_latent \
--exp_name [Wandb Run Name for Training Pretrained Host Model] \
--debug