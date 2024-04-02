from typing import Dict, List
import numpy as np

from src.vars import *
from src.embedding.abstract import DocEncoder, TextEncoder

"""
ChunkEncoder is a DocEncoder that splits the document text into several smaller chunks before 
encoding the individual chunks. This is helpful to address the mismatch in size of the query and 
the document. It also retains the embedding of the original text. 
"""
class ChunkEncoder(DocEncoder):
    def __init__(self, encoder: TextEncoder, chunksize: int, slide: int):
        self.encoder = encoder
        self.chunksize = chunksize
        self.slide = slide
    
    def encode(self, embeddings: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        res = {}
        for fname, text in embeddings.items():
            if fname in CAT_TRACKS:
                text.append(", ".join(text))
                e = np.array([self.encoder.encode(t) for t in text])
            elif fname in SEM_TRACKS:
                e = [self.encoder.encode(t) for t in text]
                if len(e) > 1:
                    # if the field has multiple entries, we also insert the average at the beginning
                    e.insert(0, np.mean(np.array(e), axis=0))
                chunked_text = sum([self._chunk(t) for t in text], []) 
                for t in chunked_text:
                    e.append(self.encoder.encode(t))
                e = np.array(e)
            else:
                continue
            res[fname] = e
        return res
    
    def _chunk(self, text: str) -> List[str]:
        stext = text.split()
        return [' '.join(stext[i:i+self.chunksize]) for i in range(0, len(stext)-self.chunksize+1, self.slide)] 