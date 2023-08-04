from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import uvicorn
from utils import torch_gc

device = "cuda:0" if torch.cuda.is_available() else "cpu"
reward_name = "/root/model/reward-model-deberta-v3-large-v2"
rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(reward_name)


class RateBody(BaseModel):
    question: str
    content: str

def rate_article(question,content):
    try:
        inputs = tokenizer(question, content, return_tensors='pt').to(device)
        score = rank_model(**inputs).logits[0].detach()
        return score.item()
    except Exception as e:
        print(traceback.format_exc())
        print('文章打分出错')
        return 0


app = FastAPI()

@app.post("/rate")
async def rate(body: RateBody, request: Request, background_tasks: BackgroundTasks):
    if (request.headers.get("Authorization") or ' ').split(" ")[1] != os.environ['MY_TOKEN']:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")
    background_tasks.add_task(torch_gc) 
    rate = rate_article(body.question, body.content)
    content = {"rate": rate}
    return JSONResponse(status_code=200, content=content)



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=6006)
