# 文件夹说明

本目录提供一个 Quick Start 数据集，开箱即用，快速体验本项目的功能

# 数据集说明

`qa_50000.jsonl` 由 huatuo-qa 数据集中采样而来，用作示例，进行了部分字段的更改
例如：

- 重命名部分字段（例如 questions -> question），规范化部分字段
- 截断过长的文本 (Milvus 对入库的文本有长度限制)
- 去除部分重复数据

eval 目录下的 `new_qa_200.jsonl` 由 `qa_50000.jsonl` 采样而来，用于评测知识库与RAG性能，并使用
`change_data.py` 脚本对 `question` 字段进行改写，模拟实际场景。

更多数据可以参见 huatuo-qa 数据集，你也可以通过 datasets 库处理其他你自己的数据集🤗

# 完整huatuo数据集

https://github.com/FreedomIntelligence/Huatuo-26M

## Streamlined version：Huatuo-Lite

提取了以下科室数据

```
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
```

保存在 `data_dir / "processed"`下

## Online Medical Encyclopedia

{
"question": "高血压的症状有哪些？",
"answer": "高血压的主要症状包括头痛、头晕、心悸..."
}

格式处理为保存在 `data_dir / "processed"`下

## 测试数据集：huatuo26M-testdatasets
