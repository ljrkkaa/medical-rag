#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理 format_data.jsonl 文件，根据 label 字段分别存储到不同的 JSONL 文件
- 截断过长文本（Milvus VARCHAR 字段限制）
- 为每个 label 生成独立的 JSONL 文件
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Milvus VARCHAR 字段的默认最大长度限制
MAX_TEXT_LENGTH = 65000
# 每个 JSONL 文件的最大记录数（现在不再分割，统一为一个大文件）
MAX_RECORDS_PER_FILE = float("inf")

# 中文标签到英文标签的映射（基于实际数据）
LABEL_MAPPING = {
    "妇产科": "obstetrics_gynecology",
    "内科": "internal_medicine",
    "皮肤性病科": "dermatology",
    "儿科": "pediatrics",
    "眼耳鼻喉科": "otolaryngology",
    "肿瘤科": "oncology",
    "神经科学": "neurology",
    "外科": "surgery",
    "男性健康科": "andrology",
    "口腔科": "dentistry",
    "心理科学": "psychology",
    "生殖健康科": "reproductive_health",
}


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    截断文本到指定最大长度

    Args:
        text: 待截断的文本
        max_length: 最大长度

    Returns:
        截断后的文本
    """
    if text is None:
        return ""
    text = str(text)
    if len(text) > max_length:
        # 截断并添加省略号提示
        return text[:max_length] + "..."
    return text


def convert_label_to_english(label: str) -> str:
    """
    将中文标签转换为英文标签

    Args:
        label: 中文标签

    Returns:
        英文标签，如果不在映射中则返回原值
    """
    return LABEL_MAPPING.get(label, label)


def process_jsonl_file(jsonl_path: str, output_dir: str) -> Dict[str, int]:
    """
    处理 JSONL 文件，根据 label 字段分别存储到不同的文件

    Args:
        jsonl_path: 输入 JSONL 文件路径
        output_dir: 输出目录

    Returns:
        {label: record_count} 字典
    """
    logger.info(f"处理文件: {jsonl_path}")

    # 按 label 分组记录
    records_by_label = {}
    total_records = 0
    skipped_records = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"行 {line_idx} JSON 解析失败: {e}")
                skipped_records += 1
                continue

            # 提取必要字段
            question = record.get("question")
            answer = record.get("answer")
            label = record.get("label", "unknown")

            # 转换标签为英文，跳过不在映射中的标签
            label = convert_label_to_english(label)
            if label not in LABEL_MAPPING.values() and label != "unknown":
                skipped_records += 1
                continue

            # 验证必要字段
            if not question or not answer:
                logger.debug(f"行 {line_idx} 缺少 question 或 answer 字段，跳过")
                skipped_records += 1
                continue

            # 截断过长文本
            question = truncate_text(str(question))
            answer = truncate_text(str(answer))

            # 构建输出记录
            output_record = {"question": question, "answer": answer}

            # 按 label 分组
            if label not in records_by_label:
                records_by_label[label] = []

            records_by_label[label].append(output_record)
            total_records += 1

    logger.info(f"读取完成: 总记录 {total_records}, 跳过 {skipped_records}")

    # 为每个 label 写入单独的文件
    label_counts = {}
    for label, records in records_by_label.items():
        output_filename = f"{label}.jsonl"
        output_file = os.path.join(output_dir, output_filename)

        write_jsonl(output_file, records)
        label_counts[label] = len(records)
        logger.info(f"已写入 {len(records)} 条记录到 {output_filename}")

    return label_counts


def write_jsonl(output_path: str, records: List[Dict]) -> None:
    """
    将记录写入 JSONL 文件

    Args:
        output_path: 输出文件路径
        records: 记录列表
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """主函数"""
    # 定义路径
    data_dir = Path(__file__).parent
    input_file = data_dir / "format_data.jsonl"
    output_dir = data_dir / "processed"

    # 检查输入文件
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理 JSONL 文件
    label_counts = process_jsonl_file(str(input_file), str(output_dir))

    logger.info("处理完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info("按 label 分组统计:")

    total_records = 0
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count} 条记录")
        total_records += count

    logger.info(f"总记录数: {total_records}")


if __name__ == "__main__":
    main()
