from pathlib import Path
import json
import argparse

DEFAULT_OUTPUT = Path("data/processed.jsonl")
DEFAULT_DATA_DIR = Path("data/raw")


def existing_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"The directory '{path_str}' does not exist or is not a directory."
        )
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a processed chunks JSONL file to use for evaluation"
    )
    parser.add_argument("-n", "--chunks", type=int, help="Chunk limit")
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_DATA_DIR,
        type=existing_dir,
        help="Path to evaluation files to generate output",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="Output file of processed chunks",
    )
    return parser.parse_args()


def generate_file(output: Path, data_dir: Path, max_chunks: int | None):
    from src.etl.chunker import DoclingHybridChunker

    chunker = DoclingHybridChunker()

    with open(output, "w", encoding="utf-8") as f:
        count = 0
        for pdf in sorted(data_dir.rglob("*.pdf")):
            for chunk in chunker.chunk(pdf):
                line = {
                    "text": chunk.contextualized_text,
                    "well": chunk.well,
                    "source": Path(chunk.source_file),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
                if max_chunks and count >= max_chunks:
                    break
            if max_chunks and count >= max_chunks:
                break


if __name__ == "__main__":
    args = parse_args()
    generate_file(args.output, args.input, args.chunks)
