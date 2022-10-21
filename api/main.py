from fastapi import FastAPI
from pydantic import BaseModel

from typing import Any, Dict, List, Union
from model.app import find_dev
from model.app import __version__ as model_version
import uvicorn

app = FastAPI()




@app.post("/")
def home(request: Dict[Any, Any]):
    l=list(request.values())
    df=find_dev(l[0])
    return {"developers":[df]}



if __name__ == '__main__':
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True,workers=2)    