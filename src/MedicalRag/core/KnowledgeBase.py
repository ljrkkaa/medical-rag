import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from ..config.models import *
from .utils import create_embedding_client
from pymilvus import (
    MilvusClient,
    DataType,
    FunctionType,
    Function,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker,
)
from pathlib import Path
from ..config.models import AppConfig
from ..embed.sparse import Vocabulary, BM25Vectorizer
from .insert import insert_rows
from copy import deepcopy
from ..embed.bm25 import BM25SparseEmbedding
from typing import Optional, Union

get_resolve_path = lambda path, file=__file__: (
    Path(file).parent / Path(path)
).resolve()

logger = logging.getLogger(__name__)

AllFields = [
    "pk",
    "text",
    "summary",
    "document",
    "source",
    "source_name",
    "domain",
    "lt_doc_id",
    "chunk_id",
    "summary_dense",
    "text_dense",
    "text_sparse",
]


class MedicalHybridKnowledgeBase:
    """医疗混合知识库 - 支持多向量字段检索"""

    def __init__(self, app_config: AppConfig):
        self.milvus_config = app_config.milvus
        self.embedding_config = app_config.embedding

        # 创建多个嵌入模型实例
        self.summary_embedding = self._create_summary_embedding()
        self.text_embedding = self._create_text_embedding()

        # 向量存储实例
        self.client = MilvusClient(
            uri=self.milvus_config.uri, token=self.milvus_config.token
        )
        self.EMBEDDERS = {
            "summary_dense": self.summary_embedding,
            "text_dense": self.text_embedding,
        }

        if self.embedding_config.text_sparse.provider == "self":
            # 如果自己管理词表，则还要创建一个BM25 Embedding
            self._vocab = Vocabulary.load(
                self.embedding_config.text_sparse.vocab_path_or_name
            )  # 词表
            # BM25 索引
            self._bm25 = BM25Vectorizer(
                vocab=self._vocab,
                domain_model=self.embedding_config.text_sparse.domain_model,
                k1=self.embedding_config.text_sparse.k1,
                b=self.embedding_config.text_sparse.b,
            )
            self.EMBEDDERS["text_sparse"] = BM25SparseEmbedding(self._vocab, self._bm25)

    def _create_summary_embedding(self) -> Embeddings:
        """创建问题嵌入模型（用于summary_dense字段）"""
        return create_embedding_client(self.embedding_config.summary_dense)

    def _create_text_embedding(self) -> Embeddings:
        """创建文本嵌入模型（用于text_dense字段）"""
        # 若两路稠密配置完全一致，则复用同一实例，避免重复占用显存/重复加载模型
        if self.embedding_config.text_dense == self.embedding_config.summary_dense:
            return self.summary_embedding
        return create_embedding_client(self.embedding_config.text_dense)

    def _ensure_collection_loaded(self, collection_name: str) -> None:
        """确保集合存在且已加载，避免检索时报 collection not loaded。"""
        if not self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"集合不存在: {collection_name}，请先执行入库脚本。")

        # 幂等调用：已加载时重复调用不会影响正确性
        try:
            self.client.load_collection(collection_name=collection_name)
        except Exception as e:
            # 常见场景：只有集合和数据，但索引尚未构建
            if "index not found" in str(e).lower():
                logger.warning("检测到索引缺失，开始自动构建索引并重试加载: %s", e)
                self.build_index()
                self.client.load_collection(collection_name=collection_name)
            else:
                raise

    def _create_collection(self):
        """使用原生 Milvus 客户端创建Collection"""
        assert (
            self.embedding_config.summary_dense.dimension
            == self.embedding_config.summary_dense.dimension
        ), "多向量单行存储时，两个嵌入模型嵌入向量维度必须相同"
        dim = self.embedding_config.summary_dense.dimension
        if self.milvus_config.drop_old:
            if self.client.has_collection(
                collection_name=self.milvus_config.collection_name
            ):
                self.client.drop_collection(
                    collection_name=self.milvus_config.collection_name
                )
            schema = MilvusClient.create_schema(
                auto_id=self.milvus_config.auto_id,  # False：你需要手动提供主键 ID（如整数或字符串）
                enable_dynamic_field=True,
            )
            if self.milvus_config.auto_id:
                schema.add_field(
                    field_name="pk", datatype=DataType.INT64, is_primary=True
                )
            else:
                schema.add_field(
                    field_name="pk",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    is_primary=True,
                )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
            )
            schema.add_field(
                field_name="summary", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="document", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="source", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="source_name", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="domain", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="lt_doc_id", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="chunk_id", datatype=DataType.INT64, max_length=65535
            )
            schema.add_field(
                field_name="summary_dense", datatype=DataType.FLOAT_VECTOR, dim=dim
            )
            schema.add_field(
                field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=dim
            )
            schema.add_field(
                field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR
            )
            if self.embedding_config.text_sparse.provider == "Milvus":
                bm25_fn = Function(
                    name="bm25_text_to_sparse",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["text_sparse"],
                )
                schema.add_function(bm25_fn)

            self.client.create_collection(
                collection_name=self.milvus_config.collection_name, schema=schema
            )

            return self.client
        else:
            return self.client  # 如果不删除老集合，那就直接返回，不要创建

    def build_index(self):
        """构建合适的索引，构建完成之后load"""
        index_params = self.client.prepare_index_params()
        # 1) 为 summary_dense 建立 HNSW 向量索引（适合高维稠密向量近似最近邻检索）
        # M：控制 HNSW 图中每个节点的最大连接数
        # efConstruction：控制索引构建时的搜索深度
        index_params.add_index(
            field_name="summary_dense",
            index_type="HNSW",
            index_name="summary_dense_index",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 200},
        )
        # 2) 为 text_dense 建立 HNSW 向量索引
        index_params.add_index(
            field_name="text_dense",
            index_type="HNSW",
            index_name="text_dense_index",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 200},
        )
        # 3) 为 text_sparse 建立稀疏索引：
        #    - provider == "self"：由本地词表/BM25 产出稀疏向量，检索时使用 IP
        #    - provider != "self"：由 Milvus BM25 Function 托管，索引 metric 使用 BM25
        if self.embedding_config.text_sparse.provider == "self":
            index_params.add_index(
                field_name="text_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                index_name="text_sparse_index",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )
        else:
            index_params.add_index(
                field_name="text_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    # DAAT_MAXSCORE：倒排检索常用策略，兼顾效率与效果
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    # BM25 超参（仅 Milvus 托管 BM25 时生效）
                    "bm25_k1": self.embedding_config.text_sparse.k1,
                    "bm25_b": self.embedding_config.text_sparse.b,
                },
            )
        # 4) 一次性创建该 collection 的全部索引
        self.client.create_index(
            collection_name=self.milvus_config.collection_name,
            index_params=index_params,
        )
        # 5) 索引创建完成后加载集合，后续才能正常查询
        self.client.load_collection(self.milvus_config.collection_name)

    @staticmethod
    def _to_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def add_documents(self, documents: List[Document]) -> int:
        """添加文档，自动处理多向量字段（已优化为批量向量化调用）"""
        rows = []

        # 1) 预处理：提取所有文本，批量 embedding（避免逐条调用，加速 3-10倍）
        summaries = []
        texts = []
        for doc in documents:
            summary = self._to_text(doc.metadata.get("summary", ""))
            text = self._to_text(doc.page_content)
            summaries.append(summary)
            texts.append(text)

        # 2) 批量调用向量化（一次性处理整个 batch，而不是逐条）
        summary_vecs = self.EMBEDDERS["summary_dense"].embed_documents(summaries)
        text_vecs = self.EMBEDDERS["text_dense"].embed_documents(texts)

        # 稀疏向量批量调用
        text_sparse_vecs = []
        if self.embedding_config.text_sparse.provider == "self":
            text_sparse_vecs = self.EMBEDDERS["text_sparse"].embed_documents(texts)

        # 3) 组织行数据
        for i, doc in enumerate(documents):
            summary = summaries[i]
            text = texts[i]
            doc.metadata["summary"] = summary
            doc.metadata["text"] = text

            # 只在第一次调用时设置向量（避免重复计算）
            if len(doc.metadata.get("summary_dense", [])) == 0:
                doc.metadata["summary_dense"] = summary_vecs[i]
            if len(doc.metadata.get("text_dense", [])) == 0:
                doc.metadata["text_dense"] = text_vecs[i]

            doc_dict = deepcopy(doc.metadata)
            filtered = {k: v for k, v in doc_dict.items() if k in AllFields}
            if not self.milvus_config.auto_id:
                filtered["pk"] = doc.metadata.get("hash_id", "")
            filtered["text"] = text

            if self.embedding_config.text_sparse.provider == "self":
                filtered["text_sparse"] = text_sparse_vecs[i]

            rows.append(filtered)

        # 4) 一次性批量插入
        insert_rows(
            client=self.client,
            collection_name=self.milvus_config.collection_name,
            rows=rows,
            show_progress=False,
        )
        return len(rows)

    def _encode_query(self, query, anns_field):
        if anns_field != "text_sparse":
            data = self.EMBEDDERS[anns_field].embed_query(query)
        else:
            if self.embedding_config.text_sparse.provider == "self":
                data = self.EMBEDDERS[anns_field].embed_query(query)  # 自己管理的词表
            else:
                data = data  # 自动托管的BM25算法时，传入的查询不需要做任何处理
        return data

    def _search(
        self,
        query: str,  # 查询的问题
        single_search_request: SingleSearchRequest,
        collection_name: str,
        output_fields: list[str],
        domain_filter: Optional[Union[str, list[str]]] = None,  # 按 domain 过滤
    ):
        """Milvus 原生的查询单个问题 https://milvus.io/docs/zh/filtered-search.md"""
        data = self._encode_query(
            query=query, anns_field=single_search_request.anns_field
        )

        # 构造过滤条件：合并用户条件 + domain 过滤
        filter_expr = single_search_request.expr or ""
        if domain_filter:
            if isinstance(domain_filter, list):
                valid_domains = [
                    d for d in domain_filter if isinstance(d, str) and d.strip()
                ]
                if valid_domains:
                    domains_expr = ", ".join([f'"{d}"' for d in valid_domains])
                    domain_cond = f"domain in [{domains_expr}]"
                else:
                    domain_cond = ""
            else:
                domain_cond = f'domain == "{domain_filter}"'
            filter_expr = (
                f"{domain_cond} && ({filter_expr})"
                if (filter_expr and domain_cond)
                else (domain_cond or filter_expr)
            )

        result = self.client.search(
            collection_name=collection_name,
            data=[data],
            filter=filter_expr,
            limit=single_search_request.limit,
            output_fields=output_fields,
            search_params={
                "metric_type": single_search_request.metric_type,
                "params": single_search_request.search_params,
            },
            anns_field=single_search_request.anns_field,
        )
        return result

    def _build_ann_search_request(
        self,
        query,
        single_search_request: SingleSearchRequest,
        domain_filter: Optional[Union[str, list[str]]] = None,
    ) -> AnnSearchRequest:
        """构建子 AnnSearchRequest 请求"""
        data = self._encode_query(
            query=query, anns_field=single_search_request.anns_field
        )

        # 构造过滤条件：合并用户条件 + domain 过滤
        filter_expr = single_search_request.expr or ""
        if domain_filter:
            if isinstance(domain_filter, list):
                valid_domains = [
                    d for d in domain_filter if isinstance(d, str) and d.strip()
                ]
                if valid_domains:
                    domains_expr = ", ".join([f'"{d}"' for d in valid_domains])
                    domain_cond = f"domain in [{domains_expr}]"
                else:
                    domain_cond = ""
            else:
                domain_cond = f'domain == "{domain_filter}"'
            filter_expr = (
                f"{domain_cond} && ({filter_expr})"
                if (filter_expr and domain_cond)
                else (domain_cond or filter_expr)
            )

        search_param = {
            "data": [data],
            "anns_field": single_search_request.anns_field,
            "param": {
                "metric_type": single_search_request.metric_type,
                "params": single_search_request.search_params,
            },
            "limit": single_search_request.limit,
            "expr": filter_expr,
        }
        return AnnSearchRequest(**search_param)

    def _hybrid_search(self, search: SearchRequest):
        """Milvus 原生混合查询 https://milvus.io/docs/zh/multi-vector-search.md"""
        anns = []
        for item in search.requests:  # 构建子查询
            anns.append(
                self._build_ann_search_request(
                    query=search.query,
                    single_search_request=item,
                    domain_filter=search.domain,
                )
            )
        if search.fuse.method == "rrf":
            rank = RRFRanker(search.fuse.k)
        elif search.fuse.method == "weighted":
            rank = WeightedRanker(*search.fuse.weights)
        result = self.client.hybrid_search(
            collection_name=search.collection_name,
            reqs=anns,
            ranker=rank,
            limit=search.limit,
            output_fields=search.output_fields,
        )
        return result

    def search(self, req: SearchRequest) -> List[Document]:
        # 运行期兜底：避免 collection 未加载导致检索失败
        self._ensure_collection_loaded(req.collection_name)

        if len(req.requests) == 1:
            # 只有一个请求搜索，走普通的search
            outputs = self._search(
                req.query,
                req.requests[0],
                req.collection_name,
                req.output_fields,
                domain_filter=req.domain,
            )[0]  # 批量中的第一条，这里先不支持批量查询
        else:
            # 有多个请求搜索，走混合search
            outputs = self._hybrid_search(req)[0]

        results = []

        for i in range(len(outputs)):  # 封装获得 List[Document]
            item = outputs[i]
            results.append(
                Document(
                    page_content=item.get("text", ""),
                    metadata={
                        "pk": item.get("pk", ""),
                        "distance": item.get("distance", 99999),
                        "chunk_id": item.get("chunk_id", -1),
                        "summary": item.get("summary", ""),
                        "document": item.get("document", ""),
                        "source": item.get("source", ""),
                        "source_name": item.get("source_name", ""),
                        "domain": item.get("domain", ""),
                        "lt_doc_id": item.get("lt_doc_id", ""),
                    },
                )
            )

        return results
