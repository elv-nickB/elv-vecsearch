from typing import Callable, Dict, List, Tuple
import torch
import threading

# field name -> list of embeddings
VectorDocument = Dict[str, List[torch.Tensor]]

# field -> list of text in that field 
TextDocument = Dict[str, List[str]]

# list of (score, uid) pairs
Result = List[Tuple[float, str]]

# function taking (query_embedding, Document) -> score
Scorer = Callable[[torch.Tensor, VectorDocument], float] 

# function taking query -> Scorer
ScorerFactory = Callable[[str, List[str]], Scorer]

# function taking (text) -> embedding
TextEmbedder = Callable[[str], torch.Tensor]

class UpdateStatus():
    def __init__(self, status: str, progress: float):
        self.status = status
        self.progress = progress
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.error = None