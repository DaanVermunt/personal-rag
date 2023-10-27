import os

import openai
import json
import hashlib

from src.secrets import secrets


def hash_inp(inp: str):
    hash_object = hashlib.sha256()  # You can choose a different hash algorithm, like sha256, sha512, md5, etc.
    hash_object.update(inp.encode('utf-8'))
    hashed_string = hash_object.hexdigest()
    return f"data/{hashed_string}"


def embedding_from_cache(inp: str):
    path = hash_inp(inp)

    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return None


def save_to_cache(inp: str, embedding: list[float]):
    with open(hash_inp(inp), 'w') as f:
        return json.dump(embedding, f)


def embed_str(inp: str) -> list[float]:
    from_cache = embedding_from_cache(inp)
    if from_cache is not None:
        return from_cache

    openai.api_key = secrets()["openai_key"]
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=inp,
        encoding_format="float"
    )

    res = response['data'][0]['embedding']
    save_to_cache(inp, res)
    return res


def create_all_embeddings():
    lines = []
    with open('../data/contract.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            if len(line) < 10:
                continue

            lines.append(line)

    embeddings = []
    for line in lines:
        embeddings.append(embed_str(line))

    with open('../data/contract_embeddings.json', 'w') as f:
        json.dump({'embeddings': embeddings, 'lines': lines}, f)


def get_embeddings():
    with open('data/contract_embeddings.json', 'r') as f:
        data = json.load(f)

    return data['embeddings'], data['lines']
