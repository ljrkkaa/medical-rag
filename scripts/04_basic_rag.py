import argparse
import importlib
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="基础RAG问答演示")
    parser.add_argument(
        "--config", type=str, default="src/MedicalRag/config/app_config.yaml"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="右下腹间断性疼痛2年，近几天加重。既往有阑尾切除术史（约2年前）。疼痛多发生于晚饭后及晨起空腹时，无明显缓解。自行口服阿莫西林效果不佳。",
        help="用户问题",
    )
    parser.add_argument(
        "--show-docs",
        action="store_true",
        help="是否展示检索到的参考资料（前3条）",
    )
    parser.add_argument(
        "--show-index",
        action="store_true",
        help="是否展示RAG分阶段索引信息（路由/merge/粗排/ColBERT）",
    )
    parser.add_argument("--domain", type=str, default=None, help="手动指定检索领域")
    parser.add_argument(
        "--routing-method",
        type=str,
        default="centroid",
        choices=["centroid", "llm"],
        help="路由方式",
    )
    parser.add_argument(
        "--routing-top-k",
        type=int,
        default=None,
        help="路由候选domain数量",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    try:
        runner_module = importlib.import_module("MedicalRag.rag.BasicRagRunner")
        basic_rag_runner_cls = getattr(runner_module, "BasicRagRunner")
        basic_rag_options_cls = getattr(runner_module, "BasicRagOptions")
        build_report = getattr(runner_module, "build_basic_rag_report")

        runner = basic_rag_runner_cls(config_path=str(config_path))
        result = runner.run(
            query=args.query,
            options=basic_rag_options_cls(
                domain=args.domain,
                routing_method=args.routing_method,
                routing_top_k=args.routing_top_k,
            ),
        )

        print(
            build_report(
                result,
                show_docs=args.show_docs,
                show_index=args.show_index,
            )
        )
    except Exception:
        logger.exception("基础RAG脚本运行失败")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
