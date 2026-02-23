import h5py
import numpy as np
import hashlib
import os
import json
import atexit
import signal


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

        self._closed = False
        self._register_cleanup_handlers()

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

        print(f'H5 stats for {embeddings_path}')
        print("shape:", self.h5_file['X'].shape)
        print("maxshape:", self.h5_file['X'].maxshape)
        print("chunks:", self.h5_file['X'].chunks)

        if not os.path.exists(self.hash_to_id_path):
            with open(self.hash_to_id_path, 'w') as hash_file:
                json.dump({}, hash_file) 
        with open(self.hash_to_id_path, 'r') as hash_file:
            self.hash_to_id = json.load(hash_file)

        self.current_size = self.h5_file["X"].shape[0]


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
        self.current_size = self.h5_file["X"].shape[0]

        for i, (seq, h) in enumerate(zip(sequences, hashes)):
            if h in self.hash_to_id:
                val = self.hash_to_id[h]
                idx = val[0] if isinstance(val, list) else val

                if idx >= self.current_size:
                    print(f"WARNING: Index {idx} out of bounds (size={self.current_size}), treating as missing")
                    embeddings.append(None)
                    missing_indices.append(i)
                    missing_seqs.append(seq)
                    continue

                #handle both formats: int (no padding) and [idx, original_dim] (padded)
                try:
                    emb = self.h5_file["X"][idx]
                except OSError as e:
                    print(f"WARNING: Corrupted data at index {idx}, removing from index: {e}")
                    self.hash_to_id.pop(h)
                    embeddings.append(None)
                    missing_indices.append(i)
                    missing_seqs.append(seq)
                    continue
                if isinstance(val, list):
                    original_dim = val[1]
                    #un-pad back to original dimension
                    emb = emb[:original_dim]
             
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
        self.current_size = self.h5_file["X"].shape[0]
        if end_idx > self.current_size:
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

        self.h5_file.flush()
        
        tmp = self.hash_to_id_path + '.tmp'
        with open(tmp, 'w') as tmp_file:
            json.dump(self.hash_to_id, tmp_file)
        os.replace(tmp, self.hash_to_id_path)

    def _register_cleanup_handlers(self):
        atexit.register(self._safe_close)
        
        # Store original handlers to restore later
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        signal.signal(signal.SIGUSR1, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, closing H5 file safely...")
        self._safe_close()
        
        # Restore original handler and re-raise
        if signum == signal.SIGINT:
            signal.signal(signum, self._original_sigint)
        else:
            signal.signal(signum, self._original_sigterm)
        os.kill(os.getpid(), signum)
    
    def _safe_close(self):
        if not self._closed and self.h5_file is not None:
            try:
                self.h5_file.flush()
                self.h5_file.close()
                print("H5 file closed safely")
            except Exception as e:
                print(f"Warning closing H5: {e}")
            finally:
                self._closed = True
                self.h5_file = None
    
    def close_h5(self):
        self._safe_close()

