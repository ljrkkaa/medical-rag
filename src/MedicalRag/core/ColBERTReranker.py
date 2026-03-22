from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.documents import Document
from pydantic import Field

logger = logging.getLogger(__name__)


class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT 风格重排器（后期交互 MaxSim）。"""

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    top_k: int = 10
    batch_size: int = 16
    device: str = "auto"

    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    torch: Any = Field(default=None, exclude=True)
    F: Any = Field(default=None, exclude=True)

    # (model_name, resolved_device) -> (tokenizer, model, torch, F)
    _MODEL_CACHE: ClassVar[Dict[Tuple[str, str], Tuple[Any, Any, Any, Any]]] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lazy_init_model()

    def _lazy_init_model(self):
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise ImportError(
                "ColBERTReranker 依赖 torch 与 transformers，请先安装。"
            ) from e

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cache_key = (self.model_name, device)
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            tokenizer, model, torch_mod, f_mod = cached
            object.__setattr__(self, "tokenizer", tokenizer)
            object.__setattr__(self, "model", model)
            object.__setattr__(self, "torch", torch_mod)
            object.__setattr__(self, "F", f_mod)
            object.__setattr__(self, "device", device)
            logger.info(
                "ColBERT模型复用缓存: model=%s, device=%s", self.model_name, device
            )
            return

        object.__setattr__(self, "torch", torch)
        object.__setattr__(self, "F", F)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoModel.from_pretrained(self.model_name)
        model.to(device)
        model.eval()
        self._MODEL_CACHE[cache_key] = (tokenizer, model, torch, F)

        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "device", device)
        logger.info("ColBERT模型加载完成: model=%s, device=%s", self.model_name, device)

    @classmethod
    def preload(
        cls, model_name: str = "bert-base-uncased", device: str = "auto"
    ) -> None:
        """按配置提前预加载 ColBERT 模型，后续实例将直接复用缓存。"""
        # 直接构造一个实例会触发 _lazy_init_model 和缓存逻辑
        cls(model_name=model_name, device=device)

    def _encode_text(self, texts: List[str]):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = self.F.normalize(embeddings, p=2, dim=-1)
        return embeddings, inputs["attention_mask"]

    def calculate_colbert_similarity(
        self,
        query_emb,
        doc_embs,
        query_mask,
        doc_masks,
    ) -> List[float]:
        """ColBERT相似度计算（MaxSim）。"""
        scores: List[float] = []
        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i : i + 1]

            similarity_matrix = self.torch.matmul(
                query_emb, doc_emb.unsqueeze(0).transpose(-2, -1)
            )
            doc_mask_expanded = doc_mask.unsqueeze(1)
            similarity_matrix = similarity_matrix.masked_fill(
                ~doc_mask_expanded.bool(), -1e9
            )

            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(
                ~query_mask_expanded.bool(), 0
            )
            colbert_score = float(max_sim_per_query_token.sum(dim=-1).item())
            scores.append(colbert_score)
        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """对文档进行ColBERT重排序。"""
        if len(documents) == 0:
            return documents

        query_embeddings, query_mask = self._encode_text([query])

        doc_texts = [doc.page_content for doc in documents]
        scored_docs = []

        for i in range(0, len(doc_texts), self.batch_size):
            batch_docs = list(documents[i : i + self.batch_size])
            batch_texts = doc_texts[i : i + self.batch_size]

            doc_embeddings, doc_masks = self._encode_text(batch_texts)
            scores = self.calculate_colbert_similarity(
                query_embeddings,
                doc_embeddings,
                query_mask,
                doc_masks,
            )

            for doc, score in zip(batch_docs, scores):
                md = dict(doc.metadata) if doc.metadata else {}
                md["colbert_score"] = score
                scored_docs.append(Document(page_content=doc.page_content, metadata=md))

        scored_docs.sort(
            key=lambda d: d.metadata.get("colbert_score", 0.0), reverse=True
        )
        return scored_docs[: self.top_k]
