from fastapi import FastAPI
from pydantic import BaseModel

from typing import Any, Dict, List, Union
from model.model import query
from model.model import __version__ as model_version
import uvicorn

app = FastAPI()




@app.post("/")
def home(request: Dict[Any, Any]):
    l=list(request.values())
    lk=list(map(int, l))
    k=query(lk)
    return {"song_names_top":k[0],"song_names_alt":k[1]}



if __name__ == '__main__':
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True,workers=2)    