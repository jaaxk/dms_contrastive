import h5py
import numpy as np
import hashlib
import os
import json
import torch

hash_to_id_path = ''
hash_to_id = {}
hash_file = None
h5_file = None

def init_h5(embeddings_path, N, embed_dim=1280):
    print(f"Initializing h5 file at {embeddings_path}")
    global hash_to_id_path
    global hash_to_id
    global hash_file
    global h5_file
    hash_to_id_path = embeddings_path + '.hash_to_id.json'

    if not os.path.exists(embeddings_path):
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        h5_file = h5py.File(embeddings_path, "w")
        h5_file.create_dataset(
                "X",
                shape=(N, embed_dim),
                dtype="float32",
                chunks=(1024, embed_dim),
            )
        h5_file.create_dataset(
                "seq_ids",
                shape=(N,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
    else:
        h5_file = h5py.File(embeddings_path, "r+")

    if not os.path.exists(hash_to_id_path):
        with open(hash_to_id_path, 'w') as hash_file:
            json.dump({}, hash_file) 
    with open(hash_to_id_path, 'r') as hash_file:
        hash_to_id = json.load(hash_file)


def seq_hash(seq):
    return hashlib.sha1(seq.encode()).hexdigest()

def load_embeddings(embeddings_path, sequences):
    """load embeddings from sequence -> hash -> h5 file
    if any are missing, return None"""

    #get hash for each sequence
    hashes = [seq_hash(seq) for seq in sequences]
    embeddings = []
    missing_seqs = []
    missing_indices = []

    for i, (seq, h) in enumerate(zip(sequences, hashes)):
        if h in hash_to_id:
            idx = hash_to_id[h]
            embeddings.append(h5_file["X"][idx])
        else:
            embeddings.append(None)
            missing_indices.append(i)
            missing_seqs.append(seq)
    
    return embeddings, missing_seqs, missing_indices

def save_embeddings(embeddings_path, sequences, embeddings):
    """save embeddings to h5 file"""
    global hash_to_id
    start_idx = max(hash_to_id.values(), default=-1) + 1
    end_idx = start_idx + len(embeddings)
    hashes = [seq_hash(seq) for seq in sequences]
    hash_to_id.update(zip(hashes, range(start_idx, end_idx)))
    #always check that indexes are unique
    assert len(set(hash_to_id.values())) == len(hash_to_id.values())
    #print(type(embeddings), embeddings.shape if isinstance(embeddings, torch.Tensor) else "not tensor")
    embeddings_np = embeddings.cpu().numpy()
    h5_file["X"][start_idx:end_idx] = embeddings_np.astype(np.float32)
    h5_file["seq_ids"][start_idx:end_idx] = hashes
    
    tmp = hash_to_id_path + '.tmp'
    with open(tmp, 'w') as tmp_file:
        json.dump(hash_to_id, tmp_file)
    os.replace(tmp, hash_to_id_path)

def close_h5():
    h5_file.close()
    