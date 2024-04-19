CUDA_VISIBLE_DEVICES="0,1,2,3" \
accelerate launch \
--main_process_port 13647 \
--num_processes 4 \
--gpu_ids 0,1,2,3 \
../main.py \
--input_path [Your Input Path] \
--save_dir [Your Save Directory] \
--train_type federated \
--algorithm fedavg \ # Choose from fedavg, fedprox, fedbn, fedpxn
--type_token \
--dpe \
--pos_enc \
--n_layers 2 \
--batch_size 64 \
--wandb_project_name [Your Wandb Project Name] \
--wandb_entity_name [Your Wandb Entity Name] \
--seed 42 \
--src_data mimiciii_cv mimiciii_mv mimiciv eicu_south eicu_west \
--mixed_precision bf16