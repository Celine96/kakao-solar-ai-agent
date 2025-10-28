from openai import OpenAI

client = OpenAI(api_key="...")

import pickle

article_chunks = [
    "냥냥몬은 고양이 포켓몬입니다.",
    "울냥몬은 냥냥몬의 진화형입니다."    
]

chunk_embeddings = [
    client.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
    for chunk in article_chunks
]

# chunk_embeddings = []
# for chunk in article_chunks:
#     embedding = client.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
#     chunk_embeddings.append(embedding)

with open("embeddings.pkl", "wb") as f:
    pickle.dump({"chunks": article_chunks, "embeddings": chunk_embeddings}, f)

