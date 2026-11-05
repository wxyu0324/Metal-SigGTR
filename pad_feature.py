import numpy as np
import torch
from tqdm import tqdm
import os


Dataset_Path = ".../datasets/"
Feature_Path = ".../output/feature"
FASTA_FILE_NAME = "protein.fa"   

ESM2_DIR = os.path.join(Feature_Path, "ESM/")  
EDSSP_DIR = os.path.join(Feature_Path, "dis/")         

OUTPUT_DIR = os.path.join(Feature_Path, "input_ESM_dis/")  
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


NODE_DIM = 2560 + 14
MAX_LEN = 3100
def prepare_features_final(pdb_id: str, sequence_str: str):
    try:
        real_len = len(sequence_str)
        emb_path = os.path.join(ESM2_DIR, f"{pdb_id}.npy")
        npz_path = os.path.join(EDSSP_DIR, f"{pdb_id}.npz")

        if not os.path.exists(emb_path) or not os.path.exists(npz_path):
            return


        seq_emb = np.load(emb_path).astype(np.float32)  
        with np.load(npz_path) as data:
            coords = data["ca_coords"]       
        
            dssp_feats = data["feat_14d"]

    
        if seq_emb.shape[0] > real_len: seq_emb = seq_emb[:real_len]
        elif seq_emb.shape[0] < real_len: seq_emb = np.pad(seq_emb, ((0, real_len - seq_emb.shape[0]), (0, 0)))

        if dssp_feats.shape[0] > real_len: dssp_feats = dssp_feats[:real_len]
        elif dssp_feats.shape[0] < real_len: dssp_feats = np.pad(dssp_feats, ((0, real_len - dssp_feats.shape[0]), (0, 0)))

        if coords.shape[0] > real_len: coords = coords[:real_len]
        elif coords.shape[0] < real_len: coords = np.pad(coords, ((0, real_len - coords.shape[0]), (0, 0)))

        node_features = np.hstack([seq_emb, dssp_feats])

        padded_X = np.zeros((MAX_LEN, 3), dtype=np.float32)
        padded_X[:real_len] = coords
        padded_X = torch.tensor(padded_X, dtype=torch.float32)

        padded_node = np.zeros((MAX_LEN, NODE_DIM), dtype=np.float32)
        padded_node[:real_len] = node_features
        padded_node = torch.tensor(padded_node, dtype=torch.float32)

        masks = np.zeros(MAX_LEN, dtype=np.int64)
        masks[:real_len] = 1
        masks = torch.tensor(masks, dtype=torch.long)

        torch.save(padded_X, os.path.join(OUTPUT_DIR, f"{pdb_id}_X.tensor"))
        torch.save(padded_node, os.path.join(OUTPUT_DIR, f"{pdb_id}_node_feature.tensor"))
        torch.save(masks, os.path.join(OUTPUT_DIR, f"{pdb_id}_mask.tensor"))

    except Exception as e:
        print(f"ID: {pdb_id} | 处理失败: {str(e)}")

def get_id_seq_map(fa_file_name: str):
    path = os.path.join(Dataset_Path, fa_file_name)
    id_map = {}
    with open(path, "rb") as f:
        lines = f.readlines()
        curr_id = None
        for line in lines:
            line_str = line.decode("utf-8").strip()
            if not line_str: continue
            if line_str.startswith(">"): curr_id = line_str[1:]
            elif curr_id:
                id_map[curr_id] = line_str
                curr_id = None
    return id_map

if __name__ == "__main__":
    train_data = get_id_seq_map(FASTA_FILE_NAME)
    for pdb_id, seq in tqdm(train_data.items()):
        prepare_features_final(pdb_id, seq)
    print("\n 所有特征合并完成！")
