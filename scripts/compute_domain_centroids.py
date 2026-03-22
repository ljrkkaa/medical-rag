"""工程化入口：调用 src.MedicalRag.rag.CentroidBuilder 构建 centroid。"""

import argparse
from pathlib import Path

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.rag.CentroidBuilder import build_and_save_centroids


def main():
    parser = argparse.ArgumentParser(description="计算 domain centroid")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="domain_centroids.npz")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--mode",
        type=str,
        default="milvus",
        choices=["milvus", "embed"],
        help="milvus: 直接用库中向量计算（推荐更快）；embed: 重新嵌入计算",
    )
    args = parser.parse_args()

    config = ConfigLoader().config
    processed_dir = Path(args.processed_dir)
    output_file = Path(args.output)

    print("=" * 60)
    print("🔧 Domain centroid 构建（工程化入口）")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Processed Dir: {processed_dir}")
    print(f"Output: {output_file}")
    print(f"Batch Size: {args.batch_size}")

    build_and_save_centroids(
        config=config,
        mode=args.mode,
        processed_dir=processed_dir,
        output_file=output_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
