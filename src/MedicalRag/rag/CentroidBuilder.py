from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Any

import numpy as np
from tqdm import tqdm

from ..config.models import AppConfig
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.utils import create_embedding_client

logger = logging.getLogger(__name__)


class DomainCentroidBuilder:
    """Domain centroid 构建器（工程化复用）。"""

    def __init__(self, config: AppConfig):
        self.config = config

    def _preferred_domain(self) -> str:
        return (self.config.routing.preferred_domain or "").strip()

    def _exclude_preferred(self, domains: list[str]) -> list[str]:
        preferred = self._preferred_domain()
        if not preferred:
            return domains
        return [d for d in domains if d != preferred]

    def _iter_domain_vectors(
        self,
        kb: MedicalHybridKnowledgeBase,
        collection_name: str,
        domain: str,
        batch_size: int,
    ):
        """迭代读取指定 domain 的 summary_dense 向量。优先 query_iterator。"""
        expr = f'domain == "{domain}"'

        if hasattr(kb.client, "query_iterator"):
            iterator = None
            try:
                iterator = kb.client.query_iterator(
                    collection_name=collection_name,
                    filter=expr,
                    output_fields=["summary_dense"],
                    batch_size=batch_size,
                )
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for row in batch:
                        vec = row.get("summary_dense")
                        if vec:
                            yield vec
                return
            except Exception as e:
                logger.warning("query_iterator 失败，回退分页 query: %s", e)
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except Exception:
                        pass

        offset = 0
        while True:
            rows = kb.client.query(
                collection_name=collection_name,
                filter=expr,
                output_fields=["summary_dense"],
                limit=batch_size,
                offset=offset,
            )
            if not rows:
                break

            for row in rows:
                vec = row.get("summary_dense")
                if vec:
                    yield vec

            got = len(rows)
            offset += got
            if got < batch_size:
                break

    def _compute_one_centroid_from_milvus(
        self,
        kb: MedicalHybridKnowledgeBase,
        collection_name: str,
        domain: str,
        batch_size: int,
    ) -> tuple[Optional[np.ndarray], int]:
        sum_vec: Optional[np.ndarray] = None
        count = 0

        for vec in self._iter_domain_vectors(kb, collection_name, domain, batch_size):
            arr = np.asarray(vec, dtype="float32")
            if sum_vec is None:
                sum_vec = np.zeros_like(arr, dtype="float64")
            sum_vec += arr
            count += 1

        if count == 0 or sum_vec is None:
            return None, 0

        centroid = (sum_vec / count).astype("float32")
        return centroid, count

    @staticmethod
    def load_domains(processed_dir: Path, fallback_domains: list[str]) -> list[str]:
        """优先从 data/processed 文件名获取 domain；目录为空时回退配置。"""
        if processed_dir.exists():
            domains = [fp.stem for fp in sorted(processed_dir.glob("*.jsonl"))]
            if domains:
                return domains
        return list(fallback_domains)

    def build_from_milvus(
        self,
        processed_dir: Path = Path("data/processed"),
        batch_size: int = 2048,
        domains: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        kb = MedicalHybridKnowledgeBase(self.config)
        collection_name = self.config.milvus.collection_name
        kb._ensure_collection_loaded(collection_name)

        if domains is None:
            domains = self.load_domains(
                processed_dir, self.config.routing.fallback_domains
            )
        domains = self._exclude_preferred(domains)

        result: dict[str, np.ndarray] = {}
        for domain in tqdm(domains, desc="计算 centroid"):
            centroid, cnt = self._compute_one_centroid_from_milvus(
                kb=kb,
                collection_name=collection_name,
                domain=domain,
                batch_size=batch_size,
            )
            if centroid is None:
                logger.warning("domain=%s 无数据，跳过", domain)
                continue
            result[domain] = centroid
            logger.info(
                "✓ %s centroid 完成: n=%s, dim=%s", domain, cnt, centroid.shape[0]
            )

        return result

    def build_from_embed(
        self,
        processed_dir: Path = Path("data/processed"),
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        if not processed_dir.exists():
            raise FileNotFoundError(f"目录不存在: {processed_dir}")

        embed_model = create_embedding_client(self.config.embedding.summary_dense)
        domain_vecs: dict[str, np.ndarray] = {}

        preferred = self._preferred_domain()
        for fp in sorted(processed_dir.glob("*.jsonl")):
            domain = fp.stem
            if preferred and domain == preferred:
                logger.info("跳过 preferred_domain 的 centroid 构建: %s", domain)
                continue
            logger.info("处理 domain: %s", domain)

            texts = []
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q = str(item.get("question", "")).strip()
                    if q:
                        texts.append(q)

            if not texts:
                logger.warning("domain %s 无有效文本", domain)
                continue

            embeddings = []
            for i in tqdm(
                range(0, len(texts), batch_size),
                desc=f"嵌入 {domain}",
                leave=False,
            ):
                batch = texts[i : i + batch_size]
                embeddings.extend(embed_model.embed_documents(batch))

            centroid = np.mean(np.asarray(embeddings, dtype="float32"), axis=0)
            domain_vecs[domain] = centroid
            logger.info("✓ %s centroid 完成, dim=%s", domain, centroid.shape[0])

        return domain_vecs

    @staticmethod
    def save_npz(domain_centroids: dict[str, np.ndarray], output_file: Path) -> None:
        if not domain_centroids:
            raise RuntimeError("未计算出任何 centroid，请检查数据或 Milvus 集合。")
        kwargs: dict[str, Any] = dict(domain_centroids)
        np.savez(str(output_file), **kwargs)

    @staticmethod
    def print_summary(
        domain_centroids: dict[str, np.ndarray], elapsed: float, output_file: Path
    ) -> None:
        print("\n✅ 完成！")
        print(f"   Domain 数: {len(domain_centroids)}")
        print(f"   Domains: {list(domain_centroids.keys())}")
        print(f"   Embedding 维度: {next(iter(domain_centroids.values())).shape[0]}")
        print(f"   保存位置: {output_file}")
        print(f"   总耗时: {elapsed:.2f}s")

        print("\n📊 验证 centroid 向量：")
        for domain, centroid in domain_centroids.items():
            print(
                f"   {domain}: shape={centroid.shape}, norm={np.linalg.norm(centroid):.4f}"
            )


def build_and_save_centroids(
    config: AppConfig,
    mode: str = "milvus",
    processed_dir: Path = Path("data/processed"),
    output_file: Path = Path("domain_centroids.npz"),
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    """便捷函数：构建并保存 centroids。"""
    builder = DomainCentroidBuilder(config)

    t0 = time.time()
    if mode == "milvus":
        domain_centroids = builder.build_from_milvus(
            processed_dir=processed_dir,
            batch_size=batch_size,
        )
    elif mode == "embed":
        domain_centroids = builder.build_from_embed(
            processed_dir=processed_dir,
            batch_size=max(1, min(batch_size, 1024)),
        )
    else:
        raise ValueError(f"不支持的 mode: {mode}")

    builder.save_npz(domain_centroids, output_file)
    builder.print_summary(
        domain_centroids, elapsed=time.time() - t0, output_file=output_file
    )
    return domain_centroids
