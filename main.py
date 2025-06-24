import argparse
import torch
from src.mywardrobe import build_index, load_index, search, encode_query
from src.mywardrobe.finetune import run_finetune

def main():
    # Print device information for debugging
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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

    try:
        if args.cmd == "prep":
            print(f"Building index from {args.image_dir} with masks from {args.mask_dir}")
            build_index(args.image_dir, args.mask_dir, args.out_emb, args.out_idx)
        elif args.cmd == "query":
            print(f"Searching for similar images to {args.query_image}")
            if args.query_text:
                print(f"With text query: '{args.query_text}'")
            search(
                args.query_image, args.query_text,
                args.top_k, args.emb_file, args.idx_file
            )
        elif args.cmd == "finetune":
            run_finetune()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure CLIP is installed: pip install openai-clip")
        print("And other dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
