from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def make_embedding(sentence: str):
    embedding = model.encode(sentence, normalize_embeddings=True)
    return embedding.tolist()

testSentence = "내일 이비인후과를 갈까 하는데 혹시 같이 내원할 사람 있나..?"

# print(make_embedding(testSentence))

emb = model.encode(testSentence)
print(emb)