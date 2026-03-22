import json
import time
from pathlib import Path
from tqdm import tqdm
import os

from MedicalRag.embed.sparse import Vocabulary, BM25Vectorizer


def iter_qa_texts(processed_dir: Path):
    """遍历 data/processed 下所有 jsonl，产出用于 BM25 的完整文本。"""
    for fp in sorted(processed_dir.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question = (
                    "" if item.get("question") is None else str(item.get("question"))
                )
                answer = "" if item.get("answer") is None else str(item.get("answer"))

                # 与入库侧 QA 文本构造保持一致
                text = f"问题: {question}\n\n答案: {answer}"
                yield text


def main():
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        raise FileNotFoundError(f"目录不存在: {processed_dir}")

    # 1. 预统计文档数
    print("📊 预统计文档数量...")
    t_count = time.time()
    total_docs_estimate = 0
    for fp in sorted(processed_dir.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            total_docs_estimate += sum(1 for _ in f if _.strip())
    print(f"   总文档数: {total_docs_estimate}")
    print(f"   耗时: {time.time() - t_count:.2f}s\n")

    # 2. 初始化词表
    print("🔤 初始化词表与分词器...")
    vocab = Vocabulary()
    vectorizer = BM25Vectorizer(vocab, domain_model="medicine")

    # 根据 CPU 核心数自动调整 workers（加速优化）
    cpu_count = os.cpu_count() or 8
    workers = min(cpu_count, 16)  # 最多 16 个 worker
    chunksize = 256  # 调大 chunksize 加速
    print(f"   Workers: {workers}, ChunkSize: {chunksize}\n")

    # 3. 分词构建
    print("⚙️  开始分词构建词表...")
    t0 = time.time()
    total_len = 0
    total_docs = 0

    texts = iter_qa_texts(processed_dir)
    pbar = tqdm(
        vectorizer.tokenize_parallel(texts, workers=workers, chunksize=chunksize),
        total=total_docs_estimate,
        desc="分词进度",
        unit="doc",
    )

    for toks in pbar:
        vocab.add_document(toks)
        total_len += len(toks)
        total_docs += 1

    pbar.close()

    # 4. 冻结与保存
    print("\n💾 冻结词表并保存...")
    t_freeze = time.time()
    vocab.freeze()
    vocab.save("vocab.pkl.gz")
    print(f"   保存耗时: {time.time() - t_freeze:.2f}s")

    t1 = time.time()

    # 5. 统计
    print("\n✅ 词表构建完成！")
    print(f"   文档数量: {total_docs}")
    print(f"   总token数量: {total_len}")
    print(f"   平均每文档token数: {total_len / total_docs:.1f}")
    print(f"   总耗时: {t1 - t0:.2f}s")
    print(f"   吞吐量: {total_docs / (t1 - t0):.1f} docs/s")


if __name__ == "__main__":
    main()
