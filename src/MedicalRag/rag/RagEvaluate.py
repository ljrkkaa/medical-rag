"""
RAG评估模块
提供基于ragas框架的RAG系统评估功能
"""

# 导入ragas评估框架的核心功能
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult

# 导入ragas评估指标：答案相关性、忠诚度、上下文召回率、上下文精确率
from ragas.metrics import AnswerRelevancy, Faithfulness, ContextRecall, ContextPrecision

# 导入抽象基类，用于定义评估器的接口
from abc import ABC, abstractmethod

# 导入ragas的LLM包装器，用于将LangChain的LLM封装为ragas可用的格式
from ragas.llms import LangchainLLMWrapper

# 导入LangChain的ChatOpenAI模型
from langchain_community.chat_models import ChatOpenAI

# 导入基础RAG组件
from .RagBase import BasicRAG

# 导入Hugging Face的Dataset类，用于处理评估数据集
from datasets import Dataset

# 导入LangChain的基础聊天模型接口
from langchain_core.language_models import BaseChatModel

# 导入ragas的LLM包装器（重复导入，可能冗余）
from ragas.llms import LangchainLLMWrapper

# 导入进度条显示库
from tqdm import tqdm

# 导入pandas用于数据处理
import pandas as pd

# 导入ragas的评估数据集类
from ragas import EvaluationDataset

# 导入ragas的嵌入包装器
from ragas.embeddings import LangchainEmbeddingsWrapper

# 导入ragas的运行配置类
from ragas.run_config import RunConfig


class RagEvaluateBase(ABC):
    """
    RAG评估器基类
    定义RAG评估的抽象接口，具体评估实现由子类完成
    """
    
    def __init__(self, rag_components: BasicRAG, eval_datasets: Dataset, eval_llm: BaseChatModel, embedding=None) -> None:
        """
        初始化评估器
        
        Args:
            rag_components: BasicRAG实例，RAG系统组件
            eval_datasets: 评估数据集，Hugging Face Dataset格式
            eval_llm: 用于评估的LLM模型
            embedding: 可选的嵌入模型，用于某些评估指标
        """
        self.rag = rag_components
        self.eval_datasets = eval_datasets
        # 将LangChain LLM包装为ragas可用的格式
        self.eval_llm = LangchainLLMWrapper(eval_llm)
        # 将LangChain嵌入模型包装为ragas可用的格式
        self.embedding = LangchainEmbeddingsWrapper(embedding)
    
    def do_sample(self, sample_data: int):
        """
        采样评测数据集
        用于从完整评估数据集中随机抽取部分数据进行快速评估
        
        Args:
            sample_data: 采样数量
        """
        # 打乱数据集并选择指定数量的样本
        self.eval_datasets = self.eval_datasets.shuffle().select(range(sample_data))
        
    @abstractmethod
    def do_evaluate(self, datasets_query_field_name: str, datasets_reference_field_name: str):
        """
        执行评估的抽象方法
        
        Args:
            datasets_query_field_name: 数据集中查询字段的名称
            datasets_reference_field_name: 数据集中参考/标准答案字段的名称
            
        Returns:
            评估结果，具体类型由子类实现决定
        """
        pass
        
            
class RagasRagEvaluate(RagEvaluateBase):
    """
    基于Ragas框架的RAG评估器实现
    使用ragas库对RAG系统进行全面评估
    """
    
    def __init__(self, rag_components: BasicRAG, eval_datasets: Dataset, eval_llm: BaseChatModel, embedding=None) -> None:
        """
        初始化Ragas评估器
        
        Args:
            rag_components: BasicRAG实例，RAG系统组件
            eval_datasets: 评估数据集
            eval_llm: 用于评估的LLM模型
            embedding: 可选的嵌入模型
        """
        super().__init__(rag_components, eval_datasets, eval_llm, embedding)
        
    def do_evaluate(self, datasets_query_field_name: str, datasets_reference_field_name: str) -> EvaluationResult:
        """
        执行RAG系统评估
        
        使用ragas框架对RAG系统进行多维度评估，包括：
        - AnswerRelevancy（答案相关性）：评估生成答案与查询的相关程度
        - Faithfulness（忠诚度）：评估答案是否基于提供的上下文
        - ContextRecall（上下文召回率）：评估检索到的上下文与标准答案的匹配程度
        - ContextPrecision（上下文精确率）：评估检索到的上下文的精确程度
        
        Args:
            datasets_query_field_name: 数据集中查询字段的名称
            datasets_reference_field_name: 数据集中参考/标准答案字段的名称
            
        Returns:
            EvaluationResult: 包含各项评估指标得分的对象
        """
        # 存储评估结果
        response_list = []
        retrieved_contexts_list = []
        queries_list = []
        references_list = []
        
        # 遍历评估数据集，对每个查询进行RAG问答
        for data_item in tqdm(self.eval_datasets, desc="Answering questions"):
            # 调用RAG系统的answer方法获取回答
            # return_document=True表示返回检索到的文档内容
            rag_response = self.rag.answer(query=data_item[datasets_query_field_name], return_document=True)
            
            # 从检索结果中提取文档内容（提取每个document的'metadata.document'字段）
            retrieved_contexts_list.append([document.metadata['document'] for document in rag_response["documents"]])
            # 提取RAG系统生成的回答
            response_list.append(rag_response["answer"])
            # 提取原始查询
            queries_list.append(str(data_item[datasets_query_field_name]))
            # 提取标准/参考答案
            references_list.append(str(data_item[datasets_reference_field_name]))
        
        # 将评估数据整理为DataFrame格式
        df = pd.DataFrame(
            {
                "user_input": queries_list,           # 用户输入/查询
                "retrieved_contexts": retrieved_contexts_list,  # 检索到的上下文
                "response": response_list,           # RAG生成的回答
                "reference": references_list,        # 参考/标准答案
            }
        )
        
        # 将DataFrame转换为ragas评估数据集格式
        rag_results = EvaluationDataset.from_pandas(df)
        
        # 执行评估，使用多个评估指标
        results = evaluate(
            dataset=rag_results,  # 评估数据集
            metrics=[
                # 答案相关性指标：需要LLM和嵌入模型
                AnswerRelevancy(llm=self.eval_llm, embeddings=self.embedding),
                # 忠诚度指标：评估答案是否忠实于检索到的上下文
                Faithfulness(llm=self.eval_llm),
                # 上下文召回率：评估检索内容与标准答案的覆盖程度
                ContextRecall(llm=self.eval_llm),
                # 上下文精确率：评估检索内容的准确程度
                ContextPrecision(llm=self.eval_llm),
            ],
            # 运行配置：设置最大工作线程数和超时时间
            run_config=RunConfig(
                max_workers=4,    # 最多4个并行工作线程
                timeout=900       # 单个任务超时时间900秒（15分钟）
            )
        )
        return results
