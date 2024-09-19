import os
import fire
import torch
import torch.nn.functional as F


def load_data(path):
    tensors = []
    for fname in os.listdir(path):
        tensors.append(
            torch.load(os.path.join(path, fname), map_location=torch.device("cpu"))
        )
    tensors = torch.cat(tensors, dim=0)
    return tensors


def compute_avg(emb):
    return torch.mean(emb, dim=0)


def compute_softmax(emb):
    return F.softmax(emb, dim=-1)


# Compute cosine similarity with avg embeddings
def cosine_sim(host_emb_avg, subj_emb_avg):
    cosine = F.cosine_similarity(host_emb_avg.unsqueeze(0), subj_emb_avg.unsqueeze(0))
    return cosine.item()


# Compute euclidean distance with avg embeddings
def euclidean(host_emb_avg, subj_emb_avg):
    euc = torch.sqrt(torch.sum((host_emb_avg - subj_emb_avg)**2))
    return euc.item()


# Compute kl divergence with average softmax
def kl_div(host_sm_avg, subj_sm_avg):
    kl = F.kl_div(host_sm_avg.log(), subj_sm_avg, reduction="batchmean")
    return kl.item()


def main(
    host_dir: str, # Directory to host embeddings 
    subj_dir: str, # Directory to candidate subject embeddings (each subject dir split by ",") 
):
    # Avg embedding computed within host server
    host = load_data(os.path.join(host_dir))
    host_emb_avg = compute_avg(host)
    host_sm_avg = compute_avg(compute_softmax(host))

    # Avg embedding/softmax computed within each candidate subject server -> each subject shares to host
    idx = 0
    subj_dict = dict()
    for dir in subj_dir.split(","): 
        subj_dict[idx] = dict()
        subj = load_data(os.path.join(dir))
        
        subj_emb_avg = compute_avg(subj)
        subj_dict[idx]["emb"] = subj_emb_avg

        subj_sm_avg = compute_softmax(subj)
        subj_dict[idx]["softmax"] = subj_sm_avg
        
        idx += 1

    # Compute cosine similarity with avg embeddings in host local server
    for idx in subj_dict.keys():
        cosine = cosine_sim(host_emb_avg, subj_dict[idx]["emb"])
        print(f"[Cosine Sim] host & candidate subject # {idx}: {cosine}")

    print("=================================")

    # Compute euclidean distance with avg embeddings in host local server
    for idx in subj_dict.keys():
        euc = euclidean(host_emb_avg, subj_dict[idx]["emb"])
        print(f"[Euclidean Dist] host & candidate subject # {idx}: {euc}")

    print("=================================")

    # Compute kl divergence with avg softmax in host local server
    for idx in subj_dict.keys():
        kl = kl_div(host_sm_avg, subj_dict[idx]["softmax"])
        print(f"[KL Div] host & candidate subject # {idx}: {kl}")


if __name__ == "__main__":
    fire.Fire(main)