
from pydantic import BaseModel


class IndexRequest(BaseModel):
    input_dir: str

class SearchRequest(BaseModel):
    q: str
    bid_id: str | None = None
    top_k: int = 5

class AskRequest(BaseModel):
    question: str
    bid_id: str | None = None
