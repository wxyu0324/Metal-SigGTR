import argparse
import re
import os
import numpy as np
from tqdm import tqdm
import torch

try:
    import esm
except Exception as e:
    raise ImportError("未找到 `esm` 包。") from e

parser = argparse.ArgumentParser()

parser.add_argument("--fasta", type=str, default=".../datasets/protein.fa")


parser.add_argument("--out_path", type=str, default=".../output/feature/ESM")

parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

fasta_file = args.fasta
output_path = args.out_path
gpu = args.gpu

ID_list = []
seq_list =[]
with open(fasta_file, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line or line.startswith(">"):
        if line.startswith(">"): ID_list.append(line[1:])
        continue
    upper = line.upper()
    if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYUBZOX]+", upper):
        seq_list.append(upper)

def load_esm2(device: torch.device):
   
    esm2_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    esm2_model = esm2_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return esm2_model, batch_converter

@torch.no_grad()
def extract_esm2_features(esm2_model, batch_converter, ids, seqs, device):
    data =[(pid, re.sub(r"[UZOB]", "X", s.upper())) for pid, s in zip(ids, seqs)]
    _, _, toks = batch_converter(data)
    toks = toks.to(device)
    out = esm2_model(toks, repr_layers=[esm2_model.num_layers], return_contacts=False)
    reps = out["representations"][esm2_model.num_layers]
    results =[]
    for i, (_, s) in enumerate(data):
        L = len(s)
        results.append(reps[i, 1 : L + 1].detach())
    return results

def main():
 
    if not os.path.exists(output_path): 
        os.makedirs(output_path)
        
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"正在加载 ESM-2 3B 模型到 {device}...")
    esm2_model, batch_converter = load_esm2(device)

    batch_size = 1
    print(f"开始提取特征，将保存至: {output_path}")
    for i in tqdm(range(0, len(ID_list), batch_size)):
        batch_ids = ID_list[i : i + batch_size]
        batch_seqs = seq_list[i : i + batch_size]
        
       
        raw_reps = extract_esm2_features(esm2_model, batch_converter, batch_ids, batch_seqs, device)
        
        for j, pid in enumerate(batch_ids):
        
            np.save(os.path.join(output_path, f"{pid}.npy"), raw_reps[j].cpu().numpy().astype(np.float16))
            
    print("全部特征提取并保存完毕！")

if __name__ == "__main__":
    main()