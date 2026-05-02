from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

router = APIRouter()

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

class SimiIn(BaseModel):
    newsent: str

testSentence = ["어제부터 감기 기운이 있는데 병원을 가봐야 하나 고민돼",
                "내일 이비인후과를 갈까 하는데 혹시 같이 내원할 사람 있나..?",
                "내일 이비인후과에 같이 갈 사람 있나요?",
                "오늘 같이 병원 갈 사람!",
                "롤 팟 1명 구함"]


emb = model.encode(testSentence)

@router.post("/modelTest")
def simitest(request: SimiIn):
    new_sentence = request.newsent

    embNew = model.encode([new_sentence])
    simi = model.similarity(embNew, emb)

    results = []

    for sentence, score in zip(testSentence, simi[0]):
        results.append({
            "sentence": sentence,
            "similarity": float(score)
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return {"new:" : new_sentence,
            "results" : results}


# similarities = model.similarity(emb, emb)
# print(similarities)

# print("exist : \n 어제부터 감기 기운이 있는데 병원을 가봐야 하나 고민돼 \n",
#         "내일 이비인후과를 갈까 하는데 혹시 같이 내원할 사람 있나..? \n",
#         "내일 이비인후과에 같이 갈 사람 있나요? \n",
#         "오늘 같이 병원 갈 사람! \n",
#         "롤 팟 1명 구함")

# newSentence = "오늘 같이 헬스장 갈 사람 있을까요"

# embNew = model.encode([newSentence])

# simi = model.similarity(embNew, emb)
# print("new : 오늘 같이 헬스장 갈 사람 있을까요")
# print(simi)

# newSentence = "칼바람 하실 분"

# embNew = model.encode([newSentence])

# simi = model.similarity(embNew, emb)
# print("new : 칼바람 하실 분")
# print(simi)

# newSentence = "병원 갈 사람"

# embNew = model.encode([newSentence])

# simi = model.similarity(embNew, emb)
# print("new : 병원 갈 사람")
# print(simi)