import argparse
from src.retrieval import build_index, search
from src.finetune import run_finetune

def main():
    parser = argparse.ArgumentParser(description="Multimodal Fashion Retrieval CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Prep: build embeddings
    p = sub.add_parser("prep")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--out_emb", default="embeddings.npy")
    p.add_argument("--out_idx", default="index_paths.txt")

    # Query
    q = sub.add_parser("query")
    q.add_argument("--query_image", required=True)
    q.add_argument("--query_text", default="", help="Text query (optional)")
    q.add_argument("--top_k", type=int, default=5)
    q.add_argument("--emb_file", default="embeddings.npy")
    q.add_argument("--idx_file", default="index_paths.txt")

    # Fine-tune
    f = sub.add_parser("finetune")

    args = parser.parse_args()

    if args.cmd == "prep":
        build_index(args.image_dir, args.mask_dir, args.out_emb, args.out_idx)
    elif args.cmd == "query":
        search(
            args.query_image, args.query_text,
            args.top_k, args.emb_file, args.idx_file
        )
    elif args.cmd == "finetune":
        run_finetune()

if __name__ == "__main__":
    main()
