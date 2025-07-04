import argparse
from pathlib import Path
import json
from PIL import Image
import torch, clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

### 🛠 `src/finetune.py`

# ── A. Environment & Paths ───────────────────────────────
DEVICE      = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {DEVICE}")

DATA_DIR    = Path("data")
CHECKPTS    = Path("checkpoints")
BLIP_MODEL  = "Salesforce/blip-image-captioning-base"
CLIP_MODEL  = "ViT-B/32"
CAP_FILE    = DATA_DIR / "captions.jsonl"
IMG_LIST    = DATA_DIR / "image_paths.txt"

# Hyperparams
EPOCHS      = 5
BATCH_SIZE  = 32
LR          = 5e-4
WD          = 1e-2
TAU         = 0.07

# ── Datasets ──────────────────────────────────────────────
class FashionCLIPDataset(Dataset):
    def __init__(self, cap_file, preprocess_fn):
        self.items = []
        with open(cap_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                self.items.append((obj["path"], obj["caption"]))
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        
        # Ensure image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)
        text_tensor = clip.tokenize([caption], truncate=True)[0]
        return img_tensor, text_tensor

def run_finetune():
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    CHECKPTS.mkdir(exist_ok=True)
    
    # Check if image list exists
    if not IMG_LIST.exists():
        print(f"Error: {IMG_LIST} not found. Please create this file with image paths.")
        return
    
    # ── Phase C: BLIP Caption Generation ────────────────────
    if not CAP_FILE.exists():
        print("Phase C: Generating captions with BLIP…")
        try:
            processor  = BlipProcessor.from_pretrained(BLIP_MODEL)
            blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(DEVICE)
            img_paths  = [p.strip() for p in open(IMG_LIST) if p.strip()]
            
            with open(CAP_FILE, "w") as out:
                for path in tqdm(img_paths, desc="Captioning images"):
                    if not os.path.exists(path):
                        print(f"Warning: Image not found: {path}")
                        continue
                    img    = Image.open(path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                    ids    = blip_model.generate(**inputs, max_new_tokens=40)
                    cap    = processor.decode(ids[0], skip_special_tokens=True)
                    out.write(json.dumps({"path": path, "caption": cap}) + "\n")
            print(f"✔ Saved captions → {CAP_FILE}")
        except Exception as e:
            print(f"Error during caption generation: {e}")
            return
    else:
        print(f"⚠ {CAP_FILE} exists; skipping caption generation.")

    # ── Phase D: CLIP Contrastive Fine-Tuning ────────────────
    print("Phase D: Fine-tuning CLIP with InfoNCE loss…")
    try:
        clip_model, clip_preprocess = clip.load(CLIP_MODEL, device=DEVICE)
        clip_model.train()

        dataset = FashionCLIPDataset(str(CAP_FILE), clip_preprocess)
        loader  = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0
        )

        optimizer = AdamW(clip_model.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS * len(loader)
        )

        for epoch in range(EPOCHS):
            total_loss = 0.0
            for imgs, txts in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                imgs = imgs.to(DEVICE)
                txts = txts.to(DEVICE)

                with torch.no_grad():
                    img_f = clip_model.encode_image(imgs)
                    txt_f = clip_model.encode_text(txts)
                    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

                logits = img_f @ txt_f.T / TAU
                labels = torch.arange(imgs.size(0), device=DEVICE)

                loss_i = torch.nn.functional.cross_entropy(logits,    labels)
                loss_t = torch.nn.functional.cross_entropy(logits.T,  labels)
                loss   = (loss_i + loss_t) / 2

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg = total_loss / len(loader)
            print(f"Epoch {epoch+1} done – avg loss: {avg:.4f}")

        ckpt_path = CHECKPTS / "clip_finetuned.pt"
        torch.save(clip_model.state_dict(), ckpt_path)
        print(f"✔ Saved fine-tuned CLIP → {ckpt_path}")
        
    except Exception as e:
        print(f"Error during CLIP fine-tuning: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    # Pass args.epochs, args.batch_size, args.checkpoint_path to your training loop
    # Example usage in your training code:
    # if args.checkpoint_path:
    #     model.load_state_dict(torch.load(args.checkpoint_path))
    # train(model, ..., epochs=args.epochs, batch_size=args.batch_size)
