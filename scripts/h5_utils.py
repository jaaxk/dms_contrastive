import h5py
import numpy as np
import hashlib
import os
import json
import torch


class EmbeddingLoader:
    

    def __init__(self, embeddings_path, N, embed_dim=1280):
        print(f"Initializing h5 file at {embeddings_path}")
        self.hash_to_id_path = embeddings_path + '.hash_to_id.json'
        self.hash_to_id = {}
        self.hash_file = None
        self.h5_file = None
        self.embeddings_path = embeddings_path
        self.N = N
        self.embed_dim = embed_dim

        if not os.path.exists(embeddings_path):
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            self.h5_file = h5py.File(embeddings_path, "w")
            self.h5_file.create_dataset(
                    "X",
                    shape=(N, embed_dim),
                    dtype="float32",
                    chunks=(1024, embed_dim),
                    maxshape=(None, embed_dim),
                )
            self.h5_file.create_dataset(
                    "seq_ids",
                    shape=(N,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    maxshape=(None,),
                )
        else:
            self.h5_file = h5py.File(embeddings_path, "r+")

        if not os.path.exists(self.hash_to_id_path):
            with open(self.hash_to_id_path, 'w') as hash_file:
                json.dump({}, hash_file) 
        with open(self.hash_to_id_path, 'r') as hash_file:
            self.hash_to_id = json.load(hash_file)


    def seq_hash(self, seq):
        return hashlib.sha1(seq.encode()).hexdigest()

    def load_embeddings(self, sequences):
        """load embeddings from sequence -> hash -> h5 file
        if any are missing, return None
        Un-pads embeddings back to original dimension if stored"""

        hashes = [self.seq_hash(seq) for seq in sequences]
        embeddings = []
        missing_seqs = []
        missing_indices = []

        for i, (seq, h) in enumerate(zip(sequences, hashes)):
            if h in self.hash_to_id:
                val = self.hash_to_id[h]
                #handle both formats: int (no padding) and [idx, original_dim] (padded)
                if isinstance(val, list):
                    idx, original_dim = val
                    emb = self.h5_file["X"][idx]
                    #un-pad back to original dimension
                    emb = emb[:original_dim]
                else:
                    idx = val
                    emb = self.h5_file["X"][idx]
                embeddings.append(emb)
            else:
                embeddings.append(None)
                missing_indices.append(i)
                missing_seqs.append(seq)
            
        #unpad if they contain '-1' - with current implementation using only gene-aware loader for one-hot, these should all have the same pad size
        
        return embeddings, missing_seqs, missing_indices

    def save_embeddings(self, sequences, embeddings):
        """save embeddings to h5 file"""
        #handle mixed format: int for unpaded (ESM), [idx, dim] for padded (one-hot)
        max_idx = max([v if isinstance(v, int) else v[0] for v in self.hash_to_id.values()], default=-1)
        start_idx = max_idx + 1
        end_idx = start_idx + len(embeddings)
        hashes = [self.seq_hash(seq) for seq in sequences]
        
        embeddings_np = embeddings.cpu().numpy()
        original_dim = embeddings_np.shape[1]
        
        # Resize datasets if needed
        current_size = self.h5_file["X"].shape[0]
        if end_idx > current_size:
            self.h5_file["X"].resize(end_idx, axis=0)
            self.h5_file["seq_ids"].resize(end_idx, axis=0)
        
        #only store dimension if padding is needed, for computational efficiency
        if original_dim < self.embed_dim:
            pad_width = self.embed_dim - original_dim
            #pad with 0s at the end
            embeddings_np = np.pad(embeddings_np, ((0, 0), (0, pad_width)), constant_values=-1)
            #store [idx, original_dim] for padded embeddings
            self.hash_to_id.update(zip(hashes, [[idx, original_dim] for idx in range(start_idx, end_idx)]))
        elif original_dim == self.embed_dim:
            #for unpaded embeddings (ESM), just store the index to maintain backward compatibility
            self.hash_to_id.update(zip(hashes, range(start_idx, end_idx)))
        else:
            raise ValueError(f"Embedding dimension {original_dim} is larger than expected {self.embed_dim}")
        
        #always check that indexes are unique
        indices = [v if isinstance(v, int) else v[0] for v in self.hash_to_id.values()]
        assert len(set(indices)) == len(indices)

        self.h5_file["X"][start_idx:end_idx] = embeddings_np.astype(np.float32)
        self.h5_file["seq_ids"][start_idx:end_idx] = hashes
        
        tmp = self.hash_to_id_path + '.tmp'
        with open(tmp, 'w') as tmp_file:
            json.dump(self.hash_to_id, tmp_file)
        os.replace(tmp, self.hash_to_id_path)

    def close_h5(self):
        self.h5_file.close()
