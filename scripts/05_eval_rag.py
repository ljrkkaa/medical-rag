"""
RAG基础评测
"""

import logging
from typing import cast
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.rag.SimpleRag import SimpleRAG
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from MedicalRag.rag.RagEvaluate import RagasRagEvaluate
from datasets import load_dataset, Dataset
from pydantic import SecretStr

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    # 加载配置
    config_manager = ConfigLoader()
    # 创建基础RAG系统
    rag = SimpleRAG(config_manager.config)
    eval_data = cast(
        Dataset,
        load_dataset(
            "json", data_files="data/eval/eval_datasets_sample10.jsonl", split="train"
        ),
    )
    qwen_llm = ChatOpenAI(
        base_url="https://www.dmxapi.cn/v1",
        model="qwen-flash-2025-07-28",
        api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY", "")),
        temperature=0.0,
        extra_body={"enable_thinking": False},
    )
    qwen_embedding = OpenAIEmbeddings(
        base_url="https://www.dmxapi.cn/v1",
        model="Qwen/Qwen3-Embedding-8B",
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    )
    eval = RagasRagEvaluate(
        rag_components=rag,
        eval_datasets=eval_data,
        eval_llm=qwen_llm,
        embedding=qwen_embedding,
    )
    eval.do_sample(3)  # 根据需要进行快速修改
    print(
        eval.do_evaluate(
            datasets_query_field_name="questions",
            datasets_reference_field_name="answers",
        )
    )


if __name__ == "__main__":
    main()







