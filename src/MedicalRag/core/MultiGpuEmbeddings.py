from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MultiGpuSentenceTransformerEmbeddings(Embeddings):
    """基于 sentence-transformers 的多GPU嵌入客户端。"""

    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = True,
        target_devices: Optional[List[str]] = None,
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name)
        self.target_devices = target_devices or ["cuda:0"]
        self.pool = self.model.start_multi_process_pool(
            target_devices=self.target_devices
        )
        logger.info(
            "Multi-GPU Embedding 已启用: model=%s, devices=%s, batch_size=%s",
            model_name,
            self.target_devices,
            batch_size,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode_multi_process(
            texts,
            pool=self.pool,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
        )
        arr = np.asarray(vecs, dtype="float32")
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype="float32").tolist()

    def close(self) -> None:
        if getattr(self, "pool", None) is not None:
            try:
                self.model.stop_multi_process_pool(self.pool)
            except Exception:
                pass
            self.pool = None

    def __del__(self):
        self.close()
