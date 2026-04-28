from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

testSentence = ["어제부터 감기 기운이 있는데 병원을 가봐야 하나 고민돼",
                "내일 이비인후과를 갈까 하는데 혹시 같이 내원할 사람 있나..?",
                "내일 이비인후과에 같이 갈 사람 있나요?",
                "오늘 같이 병원 갈 사람!"]


emb = model.encode(testSentence)
print(emb)
print(emb.shape)

similarities = model.similarity(emb, emb)
print(similarities)

newSentence = "오늘 같이 헬스장 갈 사람 있을까요"

embNew = model.encode([newSentence])

simi = model.similarity(embNew, emb)
print(simi)

# 영어 문장 테스트
"""
new_sentence = "I feel like I have a cold, and I'm looking for someone to go to the hospital with."

existing_sentences = [
    "Is anyone going to the ENT clinic tomorrow and wants to go together?",
    "Does anyone want to get chicken for dinner tonight?",
    "I think I have a cold, so I'm going to book a hospital appointment.",
    "I'm looking for someone to go to the gym with.",
    "I don't want to go to the hospital alone, so I'm looking for someone to go with me."
]

emb = model.encode(existing_sentences)
embNew = model.encode([new_sentence])

simi = model.similarity(embNew, emb)
print(simi)
""" 