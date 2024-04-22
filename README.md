# EHRFL: Federated Learning Framework for Heterogeneous EHRs and Precision-guided Selection of Participating Clients
<table align="center">
  <tr>
    <td><img src="https://github.com/ji-youn-kim/EHRFL/blob/master/resources/Figure1.png?raw=true" width="500"/></td>
    <td><img src="https://github.com/ji-youn-kim/EHRFL/blob/master/resources/Figure2.png?raw=true" width="500"/></td>
  </tr>
</table>

## Overview
In this study, we provide solutions to two practical yet overlooked scenarios in federated learning for electronic health records (EHRs): firstly, we introduce EHRFL, a framework that facilitates federated learning across healthcare institutions with distinct medical coding systems and database schemas using text-based linearization of EHRs. 
Secondly, we focus on a scenario where a single healthcare institution initiates federated learning to build a model tailored for itself, in which the number of clients must be optimized in order to reduce expenses incurred by the host. For selecting participating clients, we present a novel precision-based method, leveraging data latents to identify suitable participants for the institution.
Our empirical results show that EHRFL effectively enables federated learning across hospitals with different EHR systems. 
Furthermore, our results demonstrate the efficacy of our precision-based method in selecting 
reduced number of participating clients without compromising model performance, resulting in lower operational costs when constructing institution-specific models.
We believe this work lays a foundation for the broader adoption of federated learning on EHRs.

- Paper link: Available Soon!

## Step-by-Step Guide

<details>
  
<summary>Pre-Training a Common Model</summary>

A common pre-trained model is needed to extract latents from the host and subject datas. \
The host can train this model by (1) setting the [Accelerate](https://huggingface.co/docs/accelerate/en/index) configuration and (2) running the code as follows. \
**Accelerate Configuration:**
```
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU # You May Use Multiple GPUs
downcast_bf16: 'no'
gpu_ids: [GPU IDs] 
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: [# of GPUs]
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
**Code Script** (also located in scripts/single.sh):
```
CUDA_VISIBLE_DEVICES=[GPU IDs] \ # You May Use Multiple GPUs
accelerate launch \
--main_process_port [Port] \
--num_processes [# of GPUs] \
--gpu_ids [GPU IDs] \ # You May Use Multiple GPUs
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
--src_data [Host Data] \
--mixed_precision bf16
```

</details>

<details>

<summary>Latent Extraction</summary>

The host sends the pre-trained model to subject clients for latent extraction. \
The host and subjects each extract latents with their respective data. \
**Accelerate Configuration:**
```
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: [GPU ID] # Use a Single GPU
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
**Code Script** (also located in scripts/extract_latent.sh):
```
CUDA_VISIBLE_DEVICES=[GPU ID] \ # Use a Single GPU
accelerate launch \
--main_process_port [Port] \
--num_processes [# of GPUs] \
--gpu_ids [GPU ID] \ # Use a Single GPU
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
--src_data [Client Data to Generate Latent] \
--mixed_precision no \
--extract_latent \
--exp_name [Wandb Run Name for Training Pretrained Host Model] \
--debug
```

</details>

<details>

<summary>Precision Computation</summary>

The host uses the extracted latents to compute precision (and recall) for each host-subject pair. \
This step is necessary for selecting clients for federated learning participation. \
The script for this step is located in scripts/precision_recall.sh.
```
python ../precision_recall.py \
--data_path [Your Data Path] \ # [Root Save Directory]/latents/seed_{seed}
--host [Host Datas] \
--subjects [Subject Datas]
```

</details>

<details>
  
<summary>Federated Learning on Selected Participating Clients</summary>

The host selects participating clients by excluding clients of low precision scores. \
With the selected clients, the host may then conduct federated learning using our EHRFL framework for heterogeneous EHR modeling.

**Accelerate Configuration:**
```
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU # You May Use Multiple GPUs
downcast_bf16: 'no'
gpu_ids: [GPU IDs] 
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: [# of GPUs]
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
**Code Script** (also located in scripts/federated.sh):

```
CUDA_VISIBLE_DEVICES=[GPU IDs] \ # You May Use Multiple GPUs
accelerate launch \
--main_process_port [Port] \
--num_processes [# of GPUs] \
--gpu_ids [GPU IDs] \ # You May Use Multiple GPUs
../main.py \
--input_path [Your Input Path] \
--save_dir [Your Save Directory] \
--train_type federated \
--algorithm [Federated Learning Algorithm] \
--type_token \
--dpe \
--pos_enc \
--n_layers 2 \
--batch_size 64 \
--wandb_project_name [Your Wandb Project Name] \
--wandb_entity_name [Your Wandb Entity Name] \
--src_data [Clients involved in Federated Learning] \
--mixed_precision bf16
```

</details>

## For Reproduction
<details> 
  
<summary>Requirements</summary>

```
# Create the conda environment
conda create -y -n EHRFL python=3.10.4

# Activate the environment
source activate EHRFL

# Install required packages
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install pandas==1.4.3 \
            transformers==4.39.0 \
            accelerate==0.27.2 \
            scikit-learn==1.2.2 \
            tqdm==4.65.0 \
            wandb==0.12.21
```

</details>

<details>

<summary>Dataset Preprocessing</summary>

Our experiments use the following datasets: [MIMIC-III](https://physionet.org/files/mimiciii/1.4/), [MIMIC-IV](https://physionet.org/files/mimiciv/2.0/), [eICU](https://physionet.org/files/eicu-crd/2.0/). \
Preprocess the data with [Integrated-EHR-Pipeline](https://github.com/Jwoo5/integrated-ehr-pipeline) as follows:

```
git clone https://github.com/Jwoo5/integrated-ehr-pipeline.git
git checkout federated
```
```
python main.py --ehr mimiciii --dest [Your Output Path] --first_icu --seed 42,43,44,45,46 --mortality --long_term_mortality --los_3day --los_7day --readmission --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc
python main.py --ehr mimiciv --dest [Your Output Path] --first_icu --seed 42,43,44,45,46 --mortality --long_term_mortality --los_3day --los_7day --readmission --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc
python main.py --ehr eicu --dest [Your Output Path] --first_icu --seed 42,43,44,45,46 --mortality --long_term_mortality --los_3day --los_7day --readmission --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc
```

</details>

