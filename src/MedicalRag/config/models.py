from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus配置"""

    uri: str = "http://localhost:19530"
    token: Optional[str] = None
    collection_name: str = "medical_knowledge"
    drop_old: bool = False
    auto_id: bool = True


# =============================================================================
# 嵌入配置
# =============================================================================
class DenseConfig(BaseModel):
    """稠密向量配置"""

    provider: Literal["openai", "ollama", "embedding"] = "embedding"
    model: str
    base_url: Optional[str] = None
    env_key_name: Optional[str] = None
    proxy: Optional[str] = None
    dimension: int = 1024
    multi_gpu: bool = False
    num_gpus: int = 1
    encode_batch_size: int = 256
    preload: bool = False  # 是否在系统启动阶段提前加载并缓存该Embedding模型


class SparseConfig(BaseModel):
    provider: Literal["self", "Milvus"] = "self"
    vocab_path_or_name: str = "vocab.pkl.gz"
    algorithm: str = "BM25"
    domain_model: str = "medicine"
    k1: float = 1.5
    b: float = 0.75
    build: dict = {"workers": 8, "chunksize": 64}


# 更新嵌入配置，支持多向量
class EmbeddingConfig(BaseModel):
    summary_dense: DenseConfig
    text_dense: DenseConfig
    text_sparse: SparseConfig


# =============================================================================
# LLM 配置
# =============================================================================
class LLMConfig(BaseModel):
    """LLM配置"""

    provider: Literal["openai", "ollama"] = "ollama"
    model: str
    base_url: Optional[str] = None
    env_key_name: Optional[str] = None
    proxy: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


# =============================================================================
# 数据配置
# =============================================================================
class DataConfig(BaseModel):
    """数据配置"""

    # 字段映射
    summary_field: str = "question"
    document_field: str = "answer"
    domain_field: Optional[str] = "domain"
    source_field: Optional[str] = "source"
    source_name_field: Optional[str] = "source_name"
    ### 文档独有 ###
    lt_doc_id_field: Optional[str] = "lt_doc_id"
    chunk_id_field: Optional[str] = "chunk_id"
    ### 文档独有 ###
    default_source: Optional[str] = "qa"  # 只支持QA和文献literature
    default_source_name: Optional[str] = "huatuo"  # QA数据源名称
    default_domain: Optional[str] = "general"
    default_lt_doc_id: Optional[str] = ""
    default_chunk_id: Optional[int] = -1


# =============================================================================
# 多轮RAG对话配置
# =============================================================================
class MultiDialogueRagConfig(BaseModel):
    """多轮对话关键配置"""

    estimate_token_fun: str = "avg"
    llm_max_token: int = 1024
    max_token_threshold: float = 1.1  # 宽松阈值
    cut_dialogue_scale: int = Field(
        default=2, ge=2, description="裁切一次砍一半，必须>=2"
    )
    smith_debug: bool = False
    console_debug: bool = False
    thinking_in_context: bool = False


# =============================================================================
# Agent对话配置
# =============================================================================
class AgentConfig(BaseModel):
    """多轮对话关键配置"""

    # analysis 模式会拆解子目标分开多次检索，并验证是否符合事实,不符合事实需要重写检索
    # normal 模式下不会拆分子目标,只会重写查询后进行检索
    # fast 模式下重写查询检索后即返回,不进行验证事实
    mode: Literal["analysis", "fast", "normal"] = "analysis"
    max_attempts: int = 3  # 重复验证事实最大次数
    network_search_enabled: bool = True  # 是否启用联网搜索
    network_search_cnt: int = 10  # 开启联网搜索时，返回的数量
    auto_search_param: bool = True  # 是否开启确定搜索参数
    search_workflow_version: Literal["v1", "v2"] = "v2"  # 联网搜索工作流版本
    enable_condense_question: bool = True  # 是否启用 condense question chain
    max_ask_num: int = Field(default=2, ge=1, le=10)  # 主动追问最大轮次
    console_debug: bool = False  # Agent链路调试日志
    web_search_provider: Literal["searxng", "custom"] = "searxng"
    searx_host: str = "http://127.0.0.1:8081"
    searx_language: Optional[str] = "zh-CN"
    searx_engines: Optional[List[str]] = None
    searx_categories: Optional[Union[str, List[str]]] = None
    searx_time_range: Optional[Literal["day", "month", "year"]] = None
    searx_safe_search: int = Field(default=1, ge=0, le=2)
    searx_v2_min_results: int = Field(default=20, ge=1, le=50)
    searx_v2_max_results: int = Field(default=30, ge=1, le=100)
    searx_v2_select_top_k: int = Field(default=6, ge=1, le=10)
    searx_v2_fetch_timeout_sec: float = Field(default=5.0, ge=1.0, le=30.0)
    searx_v2_max_content_chars: int = Field(default=3000, ge=500, le=20000)

    @field_validator("searx_v2_max_results")
    @classmethod
    def validate_v2_max_results(cls, v: int, info):
        min_results = info.data.get("searx_v2_min_results", 20)
        if v < min_results:
            raise ValueError("searx_v2_max_results 不能小于 searx_v2_min_results")
        return v


# =============================================================================
# 路由配置
# =============================================================================
class RoutingConfig(BaseModel):
    method: Literal["centroid", "llm"] = "centroid"
    centroid_file: str = "domain_centroids.npz"
    preferred_domain: Optional[str] = Field(
        default="encyclopedia",
        description="检索时默认优先包含的domain（如百科库）",
    )
    always_include_preferred_domain: bool = Field(
        default=True,
        description="是否始终将 preferred_domain 加入检索 domain 列表",
    )
    top_k: int = Field(default=2, ge=1, le=5, description="返回候选domain数量")
    auto_build_centroid: bool = Field(
        default=True,
        description="当 centroid_file 不存在时，是否自动构建并保存",
    )
    centroid_build_mode: Literal["milvus", "embed"] = Field(
        default="milvus",
        description="自动构建 centroid 的模式：milvus(快) 或 embed(重算)",
    )
    centroid_processed_dir: str = Field(
        default="data/processed",
        description="自动构建时用于发现 domain 的 processed 目录",
    )
    centroid_batch_size: int = Field(
        default=2048,
        ge=1,
        description="自动构建 centroid 的批大小",
    )
    # 当 centroid 文件缺失时用于 llm 路由的候选标签
    fallback_domains: List[str] = Field(
        default_factory=lambda: [
            "andrology",
            "dentistry",
            "dermatology",
            "encyclopedia",
            "internal_medicine",
            "neurology",
            "obstetrics_gynecology",
            "oncology",
            "otolaryngology",
            "pediatrics",
            "psychology",
            "reproductive_health",
            "surgery",
        ]
    )
    # 可选：为路由单独指定一个更轻量/更便宜的LLM；不填则复用主 llm
    llm: Optional[LLMConfig] = None


# =============================================================================
# 分阶段检索与重排配置
# =============================================================================
class CoarseRankConfig(BaseModel):
    """粗排序融合分数配置"""

    weight_distance: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Milvus返回distance转换后的分数权重",
    )
    weight_rank: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="每个domain内名次分数权重",
    )
    weight_domain_prior: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="路由domain先验分数权重",
    )


class ColBERTRerankConfig(BaseModel):
    """ColBERT重排配置"""

    enabled: bool = True
    model_name: str = "bert-base-uncased"
    max_length: int = Field(default=128, ge=16, le=512)
    top_k: int = Field(default=10, ge=1, le=50)
    batch_size: int = Field(default=16, ge=1, le=256)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    preload: bool = Field(
        default=False,
        description="是否在系统初始化时预加载 ColBERT 模型到内存",
    )


class StagedRetrievalConfig(BaseModel):
    """查询 -> 路由 -> 分域检索 -> merge -> 粗排 -> ColBERT -> LLM"""

    enabled: bool = True
    per_domain_limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="每个domain检索返回条数（融合后）",
    )
    merge_limit: int = Field(
        default=60,
        ge=1,
        le=300,
        description="多domain合并后保留的候选数",
    )
    coarse_rank: CoarseRankConfig = Field(default_factory=CoarseRankConfig)
    colbert: ColBERTRerankConfig = Field(default_factory=ColBERTRerankConfig)


# =============================================================================
# 检索构建配置
# =============================================================================
class RetrievalConfig(BaseModel):
    """默认检索请求构建配置（用于 BasicRAG 初始化 SearchRequest）。"""

    use_summary_dense: bool = True
    use_text_sparse: bool = True
    dense_limit: int = Field(default=10, ge=1, le=500)
    sparse_limit: int = Field(default=10, ge=1, le=500)
    dense_ef: int = Field(default=64, ge=1, le=4096)
    sparse_drop_ratio_search: float = Field(default=0.0, ge=0.0, le=1.0)
    final_limit: int = Field(default=10, ge=1, le=50)
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=500)
    weighted_weights: List[float] = Field(default_factory=lambda: [0.8, 0.2])


# =============================================================================
# 更新主配置类
# =============================================================================
class AppConfig(BaseModel):
    milvus: MilvusConfig
    embedding: EmbeddingConfig  # 包含multi_vector配置
    llm: LLMConfig
    data: DataConfig
    multi_dialogue_rag: MultiDialogueRagConfig
    agent: AgentConfig
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    staged_retrieval: StagedRetrievalConfig = Field(
        default_factory=StagedRetrievalConfig
    )
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


# =============================================================================
# 检索时需要传入的数据模型
# =============================================================================

AnnsField = Literal["summary_dense", "text_dense", "text_sparse"]

OUTPUT_FIELDS = (
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
)
OutputFields = Literal[
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


class FusionSpec(BaseModel):
    method: Literal["rrf", "weighted"] = Field("rrf", description="向量融合策略")
    k: Optional[int] = Field(
        default=60,
        gt=0,
        le=200,
        description="如果使用rrf融合策略,那么这个k值会影响结果",
    )  # RRF常用k=60
    weights: Optional[List] = Field(
        [0.3, 0.4, 0.3],
        description="如果使用weighted融合策略,那么这个weights会影响结果",
    )


class SingleSearchRequest(BaseModel):
    anns_field: AnnsField = Field("summary_dense", description="向量检索字段")
    metric_type: Literal["COSINE", "IP"] = Field(
        "COSINE", description="向量距离计算指标,除了稀疏向量,其余都用'COSINE'"
    )
    search_params: dict = Field(
        {"ef": 64},
        description="如果是稀疏向量检索,那么应该指定drop_ratio_search,值为float,例如0.0,否则指定参数ef,值为int",
    )
    limit: int = Field(
        default=50, gt=0, le=500, description="限制这个向量检索字段返回的多少条数据"
    )
    expr: Optional[str] = Field(
        "",
        description="过滤不符合这个表达式的数据,例如当需要筛选数据源时,填入:'source == qa',一般不需要更改,除非用户指定",
    )


class SearchRequest(BaseModel):
    query: str = Field("", description="查询文本")
    collection_name: str = Field(
        default="medical_knowledge",
        description="查询的collection,默认为'medical_knowledge'",
    )
    requests: List[SingleSearchRequest] = Field(
        default_factory=list,
        description="多路向量查询的检索配置",
    )
    output_fields: List[str] = Field(
        default_factory=lambda: ["text", "summary", "document"],
        description="最后输出的参考文档字段;text是由summary和document组合而来",
    )
    fuse: Optional[FusionSpec] = Field(
        default_factory=lambda: FusionSpec(method="rrf", k=60, weights=[0.3, 0.4, 0.3]),
        description="向量融合策略",
    )
    limit: int = Field(
        default=5,
        gt=0,
        le=10,
        description="经过融合排序之后,最终返回的数据量大小,请不要大于10篇",
    )
    domain: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="按领域过滤（可选）。可传单个领域或多个领域列表",
    )


def cast_output_fields(values: List[str]) -> List[OutputFields]:
    return [v for v in values if v in OUTPUT_FIELDS]
