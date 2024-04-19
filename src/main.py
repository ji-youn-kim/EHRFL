import os, argparse, re
from accelerate import Accelerator
from accelerate.logging import get_logger
import datetime, shutil
import wandb

from trainer.federated import FedTrainer
from trainer.baseline import BaseTrainer
import utils

logger = get_logger(__name__)

os.environ["OMP_NUM_THREADS"] = "8"

available_srcs = [
    "mimiciii_mv",
    "mimiciii_cv",
    "mimiciv",
    "eicu_west",
    "eicu_south",
    "eicu_73",
    "eicu_264",
    "eicu_420",
    "eicu_338",
    "eicu_300",
]

def get_parser():
    parser = argparse.ArgumentParser()

    # checkpoint configs
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_prefix", type=str, default="checkpoint")

    # dataset
    parser.add_argument(
        "--train_type",
        choices=["single", "federated", "pooled"],
        type=str,
        default="single",
    )

    parser.add_argument(
        "--src_data",
        nargs="*",
        action="store",
        choices=available_srcs,
        type=str,
        default=available_srcs,
    )

    parser.add_argument(
        "--pred_tasks",
        default=['mortality', 'long_term_mortality', 'los_3day', 'los_7day', 'readmission', 'final_acuity', 'imminent_discharge', 'diagnosis', 'creatinine', 'bilirubin', 'platelets', 'wbc'],
        choices=['mortality', 'long_term_mortality', 'los_3day', 'los_7day', 'readmission', 'final_acuity', 'imminent_discharge', 'diagnosis', 'creatinine', 'bilirubin', 'platelets', 'wbc'],
        type=list,
        help=""
    )

    # trainer
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"]) 
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--valid_subsets", type=list, default=["valid", "test"])
    parser.add_argument("--patience", type=int, default=50)

    # model hyper-parameter configs
    parser.add_argument(
        "--eventencoder", type=str, choices=["transformer"], default="transformer"
    )
    parser.add_argument("--pred_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--type_token", action="store_true")
    parser.add_argument("--dpe", action="store_true")
    parser.add_argument("--pos_enc", action="store_true")
    parser.add_argument("--pred_pooling", choices=["cls", "mean"], default="mean")
    parser.add_argument("--map_layers", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_word_len", type=int, default=128)
    parser.add_argument("--alibi_const", type=int, default=3)
    parser.add_argument("--time_embedding", choices=['sinusoidal', 'alibi_time_sym'], default="alibi_time_sym")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "bf16", "fp16"], required=True)
    parser.add_argument("--debug", action="store_true", default=False)

    # Hyper-parameters for Federated Learning
    parser.add_argument(
        "--algorithm", type=str, default="None", choices=["fedavg", "fedprox", "fedbn", "fedpxn"]
    )
    parser.add_argument("--communications", type=int, default=300)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.001) 

    # Wandb
    parser.add_argument("--wandb_entity_name", type=str, required=True)
    parser.add_argument("--wandb_project_name", type=str, required=True)
    
    # Extracting Client Latent Vectors
    parser.add_argument("--extract_latent", action="store_true", help="whether to save data tensors from trained model for precision computation")
    parser.add_argument("--exp_name", type=str, help="Wandb run name for the pretrained model (host model) to extract latents from. Only include this argument when extracting client latent vectors.")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    args.src_data = list(sorted(args.src_data))
    args.pred_tasks = [utils.get_task(task) for task in args.pred_tasks]
    
    os.makedirs(args.save_dir, exist_ok=True)

    if args.extract_latent:
        assert args.exp_name

        api = wandb.Api()
        run = api.run(f'{args.wandb_entity_name}/{args.wandb_project_name}/{args.exp_name}')
        
        assert args.seed == run.config['seed']
        args.subject = args.src_data[0]
        
        available_clients = "|".join(available_srcs)
        pattern = f"({'|'.join(available_srcs)})$"
        host = re.search(pattern, args.exp_name).group() # extract host from wandb run name (data trained to build the pretrained model)
        latent_save_dir = f'{args.save_dir}/latents/seed_{args.seed}/{host}/{args.subject}'
        args.latent_save_dir = latent_save_dir
        if os.path.exists(args.latent_save_dir) and os.path.isdir(args.latent_save_dir):
            shutil.rmtree(args.latent_save_dir)
            print(f"Removing and re-generating folder {args.latent_save_dir}...")
        os.makedirs(args.latent_save_dir, exist_ok=True)
    else:
        if args.train_type == 'federated':
            args.exp_name = "_".join(
                [
                    datetime.datetime.today().strftime("%m%d_%H%M"),
                    args.algorithm,
                    str(args.seed),
                    "_".join(args.src_data),
                ]
            )
        else:
            args.exp_name = "_".join(
            [
                datetime.datetime.today().strftime("%m%d_%H%M"),
                args.train_type,
                str(args.seed),
                "_".join(args.src_data),
            ]
        )

    os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)

    # Initialize Accelerator
    logging_method = None if args.debug else "wandb"
    accelerator = Accelerator(
        log_with=logging_method, split_batches=True, mixed_precision=args.mixed_precision
    )
    if not args.debug:
        accelerator.init_trackers(
            project_name=args.wandb_project_name,
            config=args,
            init_kwargs={
                "wandb": {
                    "entity": args.wandb_entity_name,
                    "reinit": True,
                    "id": args.exp_name
                }
            }
        )

    logger.info("Start Training")

    if args.train_type == "federated":
        trainer = FedTrainer(args, accelerator)
    elif args.train_type in ["single", "pooled"]:
        trainer = BaseTrainer(args, accelerator)
    else:
        raise NotImplementedError()
    
    trainer.train()

    logger.info("Done Training")
