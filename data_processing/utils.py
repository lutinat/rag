import json

def save_chunks_jsonl(chunks, path):
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

def load_chunks_jsonl(path):
    chunks = []
    with open(path, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks