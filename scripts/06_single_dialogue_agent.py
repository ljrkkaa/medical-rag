from MedicalRag.agent.SearchGraph import SearchGraph
import logging
from MedicalRag.agent.tools import searxng_search
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.utils import create_llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    config_manager = ConfigLoader()
    # config_manager.change({
    #     "llm.model":"qwen3:32b",
    #     "agent.network_search_cnt": 5
    # })
    # 使用配置文件中的 llm，避免硬编码 Tongyi 导致 InvalidApiKey
    power_model = create_llm_client(config_manager.config.llm)
    graph = SearchGraph(
        config_manager.config, power_model=power_model, websearch_func=searxng_search
    )
    try:
        result = graph.answer("宝宝出生20天了晚上卧室空调开太高对黄疸有影响吗？")
        print(result)
    except KeyError as e:
        logger.error("缺少环境变量: %s", e)
        logger.error(
            "请先配置 app_config.yaml 中 llm.env_key_name 对应的环境变量（当前是 DEEPSEEK_API_KEY）"
        )
        raise
