import numpy as np

def load_vec(path):
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            key = parts[0]
            vec = np.array(list(map(float, parts[1:])))
            embeddings[key] = vec
    return embeddings

def avg_entity_vec(entity_list, embedding_dict):
    vecs = [embedding_dict[e] for e in entity_list if e in embedding_dict]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)
