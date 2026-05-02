from fastapi import FastAPI
from app.model import testModel

app = FastAPI()
app.include_router(testModel.router)

@app.get("/")
def read_root():
    return {"hello" : "world"}