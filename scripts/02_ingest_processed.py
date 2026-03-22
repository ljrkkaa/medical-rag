import json
import logging
from pathlib import Path

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.IngestionPipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_processed_records(processed_dir: Path) -> list[dict]:
    """读取 processed 目录下所有 jsonl，并按文件名注入 domain。"""
    jsonl_files = sorted(processed_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"未找到数据文件: {processed_dir}/*.jsonl")

    records: list[dict] = []
    for fp in jsonl_files:
        domain = fp.stem
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("跳过非法 JSON 行: %s:%s", fp, line_no)
                    continue

                # 单 collection + 多 domain 元数据
                item.setdefault("domain", domain)
                item.setdefault("source", "qa")
                item.setdefault("source_name", domain)
                records.append(item)

    logger.info("读取完成: files=%s, records=%s", len(jsonl_files), len(records))
    return records


def main() -> None:
    config_manager = ConfigLoader()
    processed_dir = Path("data/processed")

    records = load_processed_records(processed_dir)

    print("配置加载成功")
    print(f"  集合名称: {config_manager.config.milvus.collection_name}")
    print(f"  记录总数: {len(records)}")

    print("\n=== 数据入库 ===")
    pipeline = IngestionPipeline(config_manager.config)
    success = pipeline.run(records)

    if not success:
        print("❌ 入库失败")
        return
    print("✅ 入库完成")


if __name__ == "__main__":
    main()
