import argparse
import json
import random
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.utils import create_embedding_client


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_centroids(npz_path: Path):
    data = np.load(npz_path)
    domains = list(data.files)
    mat = np.stack([data[d] for d in domains]).astype("float32")
    mat = l2_normalize(mat)
    return domains, mat


def _reservoir_sample_questions(fp: Path, k: int, rng: random.Random):
    """对单个 jsonl 做流式 reservoir sampling，避免把全量问题读入内存。"""
    if k <= 0:
        return []

    reservoir = []
    seen = 0

    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            q = str(obj.get("question", "")).strip()
            if not q:
                continue

            seen += 1
            if len(reservoir) < k:
                reservoir.append(q)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = q

    return reservoir


def load_samples(
    processed_dir: Path,
    per_domain: int,
    seed: int,
    allowed_domains: set[str] | None = None,
):
    rng = random.Random(seed)
    samples = []  # (true_domain, question)
    skipped_domains = []

    for fp in sorted(processed_dir.glob("*.jsonl")):
        true_domain = fp.stem
        if allowed_domains is not None and true_domain not in allowed_domains:
            skipped_domains.append(true_domain)
            continue

        selected = _reservoir_sample_questions(fp, per_domain, rng)
        samples.extend((true_domain, q) for q in selected)

    return samples, skipped_domains


def main():
    parser = argparse.ArgumentParser(description="快速评估 domain centroid 路由准确率")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--centroid-file", type=str, default="domain_centroids.npz")
    parser.add_argument(
        "--config", type=str, default="src/MedicalRag/config/app_config.yaml"
    )
    parser.add_argument(
        "--per-domain", type=int, default=5000, help="每个 domain 抽样数"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="embedding 批大小")
    parser.add_argument("--topk", type=int, default=3, help="统计 top-k 命中率")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    centroid_file = Path(args.centroid_file)

    if not processed_dir.exists():
        raise FileNotFoundError(f"processed 目录不存在: {processed_dir}")
    if not centroid_file.exists():
        raise FileNotFoundError(f"centroid 文件不存在: {centroid_file}")

    print("=" * 70)
    print("🚀 Domain Routing Fast Eval")
    print("=" * 70)

    t0 = time.time()

    # 1) 载入 centroid
    domain_names, centroid_mat = load_centroids(centroid_file)
    print(f"Domain 数: {len(domain_names)} -> {domain_names}")
    domain_set = set(domain_names)

    topk = max(1, min(args.topk, len(domain_names)))
    if topk != args.topk:
        print(f"[Warn] topk={args.topk} 超出范围，自动调整为 {topk}")

    # 2) 抽样问题
    samples, skipped_domains = load_samples(
        processed_dir, args.per_domain, args.seed, allowed_domains=domain_set
    )
    if not samples:
        raise RuntimeError("没有可评估样本")
    if skipped_domains:
        print(
            f"[Warn] 以下 domain 在 centroid 文件中不存在，已跳过: {sorted(skipped_domains)}"
        )

    true_domains = [d for d, _ in samples]
    queries = [q for _, q in samples]
    print(f"样本总数: {len(samples)} (每域最多 {args.per_domain})")

    # 3) 初始化 embedding 模型（只加载一次）
    cfg = ConfigLoader(args.config).config
    embedder = create_embedding_client(cfg.embedding.summary_dense)

    # 4) 批量 embedding（比逐条快）
    all_vecs = []
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Embedding"):
        batch = queries[i : i + args.batch_size]
        vecs = embedder.embed_documents(batch)
        all_vecs.extend(vecs)

    query_mat = np.array(all_vecs, dtype="float32")
    query_mat = l2_normalize(query_mat)

    # 5) 相似度矩阵：Q x D
    sims = query_mat @ centroid_mat.T

    # 6) 指标统计
    top1_idx = np.argmax(sims, axis=1)
    topk_idx = np.argsort(-sims, axis=1)[:, :topk]

    correct_top1 = 0
    correct_topk = 0
    per_domain = defaultdict(lambda: {"n": 0, "top1": 0, "topk": 0})
    conf = defaultdict(Counter)

    for i, true_d in enumerate(true_domains):
        pred1 = domain_names[top1_idx[i]]
        predk = {domain_names[j] for j in topk_idx[i]}

        per_domain[true_d]["n"] += 1
        conf[true_d][pred1] += 1

        if pred1 == true_d:
            correct_top1 += 1
            per_domain[true_d]["top1"] += 1
        if true_d in predk:
            correct_topk += 1
            per_domain[true_d]["topk"] += 1

    total = len(samples)
    print("\n=== Overall ===")
    print(f"Top-1 Acc: {correct_top1}/{total} = {correct_top1 / total:.4f}")
    print(f"Top-{topk} Acc: {correct_topk}/{total} = {correct_topk / total:.4f}")

    print("\n=== Per-domain ===")
    for d in sorted(per_domain.keys()):
        n = per_domain[d]["n"]
        a1 = per_domain[d]["top1"] / n if n else 0
        ak = per_domain[d]["topk"] / n if n else 0
        print(f"{d:20s} n={n:4d}  top1={a1:.4f}  top{topk}={ak:.4f}")

    print("\n=== Confusion (top-3) ===")
    for d in sorted(conf.keys()):
        print(f"{d:12s} {conf[d].most_common(3)}")

    print(f"\n耗时: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
