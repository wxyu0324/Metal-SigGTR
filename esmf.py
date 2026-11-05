import torch
import esm
import numpy as np
import os
import gc
from pathlib import Path
from timeit import default_timer as timer
import biotite.structure as struc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

AA_ORDER = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ORDER)}
MAX_SASA = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
            185, 160, 145, 180, 225, 115, 140, 155, 255, 230]
AA_TO_MAX_SASA = {aa: MAX_SASA[idx] for idx, aa in enumerate(AA_ORDER)}

def read_3line_fasta(path: str):
    sequences = {}
    if not Path(path).exists(): return sequences 
    with open(path, "r", encoding="utf-8") as f: 
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 3): 
            if i + 1 < len(lines): 
                header = lines[i] 
                if header.startswith('>'): 
                    sid = header[1:].split()[0] 
                    seq = lines[i+1].upper() 
                    clean_seq = ''.join([aa for aa in seq if aa in AA_ORDER]) 
                    if clean_seq: sequences[sid] = clean_seq 
    return sequences

def get_model_on_gpu(gpu_id):
    print(f"正在加载 ESMFold 到 GPU {gpu_id}...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def esmfold_predict_core(sequence, model, device):
    seq_len = len(sequence)
    
   
    if seq_len > 1500:
        model.set_chunk_size(1) 
    else:
        model.set_chunk_size(64)

    try:
        torch.cuda.empty_cache()
        gc.collect()
        with torch.inference_mode():
            
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                output = model.infer(sequence, num_recycles=1)
        return _process_output(output, seq_len)

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print(f" [提示] GPU 显存依然不足(len={seq_len})，正在切换到 CPU 模式（强制 Float32）...")
            torch.cuda.empty_cache()
            
           
            model = model.to("cpu").float() 
            
            with torch.inference_mode():
              
                output = model.infer(sequence, num_recycles=1)
           
            model = model.to(device) 
            return _process_output(output, seq_len)
        else:
            raise e

def _process_output(output, expected_len):
    full_atom_coords = output["positions"].detach().cpu().numpy()
    if full_atom_coords.ndim == 5: full_atom_coords = full_atom_coords[-1, 0, :, :, :]
    elif full_atom_coords.ndim == 4: full_atom_coords = full_atom_coords[0, :, :, :]
    if full_atom_coords.shape[0] > expected_len: full_atom_coords = full_atom_coords[:expected_len]
    ca_coords = full_atom_coords[:, 1, :]
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
    return ca_coords, dist_mat, full_atom_coords


def calculate_standard_rsa(ca_coords, sequence):
    ca_coords = np.asarray(ca_coords); n_res = len(ca_coords)
    rsa = np.zeros(n_res); seq_to_use = sequence[:n_res]; res_names = list(seq_to_use)
    AA_1TO3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    res_names_3letter = [AA_1TO3.get(aa, 'GLY') for aa in res_names]
    atom_array = struc.AtomArray(n_res)
    atom_array.coord = ca_coords; atom_array.atom_name = ["CA"]*n_res; atom_array.res_name = res_names_3letter; atom_array.res_id = np.arange(1, n_res+1)
    try:
        sasa = struc.sasa(atom_array, probe_radius=1.4)
        for i in range(n_res):
            max_sasa = AA_TO_MAX_SASA.get(res_names[i], 100.0)
            rsa[i] = min(1.0, max(0.0, sasa[i] / max_sasa))
    except: pass
    return rsa

def calculate_dihedral(a, b, c, d):
    b1,b2,b3 = b-a, c-b, d-c
    n1 = np.cross(b1,b2); n2 = np.cross(b2,b3)
    n1 /= (np.linalg.norm(n1)+1e-10); n2 /= (np.linalg.norm(n2)+1e-10)
    return np.degrees(np.arctan2(np.dot(np.cross(n1, n2), b2/(np.linalg.norm(b2)+1e-10)), np.dot(n1, n2)))

def get_dihedrals_from_coords(full_atom_coords, sequence):
    n_res = full_atom_coords.shape[0]; phi = np.zeros(n_res); psi = np.zeros(n_res)
    for i in range(n_res):
        try:
            if i > 0: phi[i] = calculate_dihedral(full_atom_coords[i-1, 2], full_atom_coords[i, 0], full_atom_coords[i, 1], full_atom_coords[i, 2])
            if i < n_res - 1: psi[i] = calculate_dihedral(full_atom_coords[i, 0], full_atom_coords[i, 1], full_atom_coords[i, 2], full_atom_coords[i+1, 0])
        except: pass
    return phi, psi

def get_one_hot_encoding(sequence, dim=11):
    n_res = len(sequence); one_hot = np.zeros((n_res, dim))
    for i, aa in enumerate(sequence):
        idx = AA_TO_IDX.get(aa, 10); one_hot[i, min(idx, 10)] = 1.0
    return one_hot

def get_sin_cos_features(sequence):
    n_res = len(sequence); pos = np.arange(n_res)
    return np.hstack([np.sin(2*np.pi*pos/n_res).reshape(-1,1), np.cos(2*np.pi*pos/n_res).reshape(-1,1)])

def build_14d_features(sequence, ca_coords):
    n_res = len(ca_coords); seq_to_use = sequence[:n_res]
    return np.hstack([calculate_standard_rsa(ca_coords, seq_to_use).reshape(-1,1), get_one_hot_encoding(seq_to_use), get_sin_cos_features(seq_to_use)])


def process_fasta_resume(fasta_path, output_dir):
    all_sequences = read_3line_fasta(fasta_path)
    if not all_sequences: return
    out_path = Path(output_dir); out_path.mkdir(parents=True, exist_ok=True)
    existing_ids = {p.stem for p in out_path.glob("*.npz")}
    to_process = [(sid, seq) for sid, seq in all_sequences.items() if sid not in existing_ids]
    to_process.sort(key=lambda x: len(x[1]))
    
    print(f"总序列数: {len(all_sequences)}, 待提取: {len(to_process)}")
    if not to_process: return

    model, device = get_model_on_gpu(0) 

    for seq_id, seq in to_process:
        print(f"\n正在处理 [{seq_id}] | 长度: {len(seq)}")
        try:
            start_t = timer()
            ca_coords, dist_mat, full_atom_coords = esmfold_predict_core(seq, model, device)
            feat_14d = build_14d_features(seq, ca_coords)
            phi, psi = get_dihedrals_from_coords(full_atom_coords, seq)
            
            min_len = len(ca_coords)
            np.savez(out_path / f"{seq_id}.npz", 
                     seq_id=seq_id, sequence=seq[:min_len], 
                     distance_matrix=dist_mat[:min_len,:min_len], 
                     feat_14d=feat_14d[:min_len], phi=phi[:min_len], 
                     psi=psi[:min_len], ca_coords=ca_coords[:min_len])
            print(f" 成功! 用时: {timer()-start_t:.1f}s")
        except Exception as e:
            print(f" [跳过] {seq_id} 失败: {str(e)}")
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    FASTA_PATH = ".../protein.fa"
    OUTPUT_DIR = ".../feature/dis/"
    process_fasta_resume(FASTA_PATH, OUTPUT_DIR)