# Client-Centered Federated Learning for Heterogeneous EHRs: Use Fewer Participants to Achieve the Same Performance
<table align="center">
  <tr>
    <td><img src="https://github.com/ji-youn-kim/EHRFL/blob/master/resources/Figure1.png?raw=true" width="500"/></td>
    <td><img src="https://github.com/ji-youn-kim/EHRFL/blob/master/resources/Figure2.png?raw=true" width="600"/></td>
  </tr>
</table>

## Overview
The increasing volume of electronic health records (EHRs) presents the opportunity to improve the accuracy and robustness of models in clinical prediction tasks. 
Unlike traditional centralized approaches, federated learning enables training on data from multiple institutions while preserving patient privacy and complying with regulatory constraints.
However, most federated learning research focuses on building a global model to serve multiple clients, overlooking the practical need for a client-specific model.
In this work, we introduce EHRFL, a federated learning framework using EHRs, designed to develop a model tailored to a single client (i.e., healthcare institution).
Our framework addresses two key challenges: (1) enabling federated learning across clients with heterogeneous EHR systems using text-based EHR modeling, and (2) reducing the cost of federated learning by selecting suitable participating clients using averaged patient embeddings.
Our experiment results on multiple open-source EHR datasets demonstrate the effectiveness of EHRFL in addressing the two challenges, establishing it as a practical solution for building a client-specific model in federated learning.

- Paper link: [Client-Centered Federated Learning for Heterogeneous EHRs: Use Fewer Participants to Achieve the Same Performance](http://arxiv.org/abs/2404.13318)

## Step-by-Step Guide

<details>
  
<summary>Pre-Training a Common Model</summary>

A common pre-trained model is needed to extract patient embeddings from the host (i.e., the client initiating federated learning) and candidate subject (i.e., the candidate client participating in federated learning alongside the host) datas. \
The host can train this model by (1) setting the [Accelerate](https://huggingface.co/docs/accelerate/en/index) configuration and (2) running the code as follows. 

**Accelerate Config:**
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
**Code Script** (scripts/single.sh):
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

<summary>Patient Embedding Extraction</summary>

The host sends the pre-trained model to each candidate subject for patient embedding extraction. \
The host and each candidate subject extracts patient embeddings with their respective data. 

**Accelerate Config:**
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
**Code Script** (scripts/extract_latent.sh):
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

<summary>Similarity Computation using Averaged Patient Embeddings</summary>

The host and each candidate subject generates an averaged patient embedding by averaging their respective patient embeddings. \
Each candidate subject sends their averaged patient embedding to the host. \
The host uses the averaged patient embeddings to compute host-subject similarity for each candidate subject. \
This step is necessary for selecting clients for federated learning participation. 

**Code Script** (scripts/similarity.sh)
```
python ../similarity.py \
--host_dir [Directory to Host Embeddings] \
--subj_dir [Directory to Subject Embeddings in Comma Separated List]
```

</details>

<details>
  
<summary>Text-based EHR Federated Learning for the Host with Selected Subjects</summary>

The host selects participating subjects by excluding candidate subjects of low similarity scores or large distances with the host. \
With our framework, the host may start building their client-specific model by conducting federated learning with the selected clients, using text-based EHR federated learning to enable training across clients of heterogeneous EHR systems.

**Accelerate Config:**
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
**Code Script** (scripts/federated.sh):

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
--src_data [s involved in Federated Learning] \
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

# For main code
pip install pandas==1.4.3 transformers==4.39.0 accelerate==0.27.2 scikit-learn==1.2.2 tqdm==4.65.0 fire==0.5.0 wandb==0.12.21

# For dataset preprocessing
pip install numpy==1.22.3 treelib==1.6.1 pyspark==3.3.1
```

</details>

<details>

<summary>Dataset Preprocessing</summary>

Our experiments use the following datasets: [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV](https://physionet.org/content/mimiciv/2.0/), [eICU](https://physionet.org/content/eicu-crd/2.0/). \
Preprocess the data with [Integrated-EHR-Pipeline](https://github.com/Jwoo5/integrated-ehr-pipeline) as follows:

```
git clone https://github.com/Jwoo5/integrated-ehr-pipeline.git
git checkout federated
```
```
python main.py --ehr mimiciii --dest [Your Output Path] --first_icu --seed 42,43,44,45,46 --mortality --long_term_mortality --los_3day --los_7day --readmission --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc
python main.py --ehr mimiciv --dest [Your Output Path] --first_icu --seed 42,43,44,45,46 --mortality --long_term_mortality --los_3day --los_7day --readmission --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc
python main.py --ehr eicu --dest [Your Output Path] --first_icu --seed 42,43,44,45,46 --mortality --long_term_mortality --los_3day --los_7day --readmission --final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc

# For cohort split
python ehrs/federated.py --dest [Your Output Path]
```

</details>

## Citation
```
@misc{kim2025clientcenteredfederatedlearningheterogeneous,
      title={Client-Centered Federated Learning for Heterogeneous EHRs: Use Fewer Participants to Achieve the Same Performance}, 
      author={Jiyoun Kim and Junu Kim and Kyunghoon Hur and Edward Choi},
      year={2025},
      eprint={2404.13318},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.13318}, 
}
```
