#!/usr/bin/env python3
"""
scripts/validate_attributes.py

Usage examples:

# 1) Evaluate using an existing classifier checkpoint:
python scripts/validate_attributes.py \
  --mapping data/forgeries/mapping.json \
  --forgery-root data/forgeries \
  --attr-list data/list_attr_celeba.txt \
  --checkpoint path/to/attr_classifier.pt \
  --out-dir results/validation

# 2) If you don't have a checkpoint, train a tiny classifier first (slow):
python scripts/validate_attributes.py \
  --mapping data/forgeries/mapping.json \
  --forgery-root data/forgeries \
  --attr-list data/list_attr_celeba.txt \
  --train \
  --processed-root data/processed \
  --epochs 5 \
  --out-dir results/validation

Outputs:
 - results/validation/mapping_validated.csv  (rows: real,forged,attr,key,...,attr_match,confidence)
 - results/validation/summary.txt
"""

import os
import json
import csv
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --------------- Utilities ---------------

def read_celeba_attr(attr_file):
    """
    Parse list_attr_celeba.txt (CelebA format).
    Returns dict: filename -> {attr_name: 0/1}
    """
    with open(attr_file, 'r') as f:
        lines = f.read().strip().splitlines()
    header = lines[1].split()  # attr names
    result = {}
    for line in lines[2:]:
        toks = line.split()
        fname = toks[0]
        vals = list(map(int, toks[1:]))
        result[fname] = {h: v for h, v in zip(header, vals)}
    return result, header

# Dataset to load images for the classifier
class ImageAttrDataset(Dataset):
    def __init__(self, root_dir, manifest, attrs, transform=None):
        """
        root_dir: directory that contains processed images organized by identity OR flat
        manifest: list of (filename, attr_vector) tuples where filename is relative path to image
        attrs: list of attribute names (order)
        """
        self.root_dir = root_dir
        self.items = manifest
        self.attrs = attrs
        self.transform = transform or transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fname, attr_vec = self.items[idx]
        img = Image.open(os.path.join(self.root_dir, fname)).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(attr_vec, dtype=torch.float32)
        return img, label, fname

# Small multilabel classifier wrapping a pretrained backbone
class AttrClassifier(nn.Module):
    def __init__(self, n_attrs, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            self.net = models.resnet18(pretrained=pretrained)
            in_f = self.net.fc.in_features
            self.net.fc = nn.Linear(in_f, n_attrs)
        else:
            # fallback
            self.net = models.resnet18(pretrained=pretrained)
            in_f = self.net.fc.in_features
            self.net.fc = nn.Linear(in_f, n_attrs)

    def forward(self, x):
        return self.net(x)

# --------------- Main logic ---------------

def load_mapping(mapping_path):
    """
    Supports either mapping.json produced earlier (list of records),
    or mapping.csv with columns that include 'real' and forged names.
    Returns list of dicts with keys: real, forgeries(list), prompt(optional), attrs(optional)
    """
    if mapping_path.lower().endswith('.json'):
        with open(mapping_path,'r') as f:
            mapping = json.load(f)
        # normalize: each record -> {real:..., forgeries:[...], attrs: {...} }
        out = []
        for rec in mapping:
            # if mapping was saved as dicts; handle flexible schemas
            real = rec.get('real') or rec.get('real_filename') or rec.get('source') 
            forges = rec.get('forged') or rec.get('forgeries') or rec.get('forged_filenames') or []
            attrs = rec.get('attrs') or rec.get('attributes') or rec.get('attr_tokens') or None
            out.append({'real': real, 'forgeries': forges, 'attrs': attrs, 'meta': rec})
        return out
    else:
        # CSV path
        out = {}
        with open(mapping_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        # group by real
        grouped = {}
        for r in rows:
            real = r['real']
            f = r['forged']
            grouped.setdefault(real, []).append({'forged': f, 'meta': r})
        res = []
        for real, items in grouped.items():
            forges = [i['forged'] for i in items]
            res.append({'real': real, 'forgeries': forges, 'attrs': None, 'meta': items})
        return res

def build_manifest_for_training(processed_root, attr_map, required_attrs):
    """
    Build a simple flat manifest: scan processed_root recursively for images,
    and return list of (relpath, attr_vector) for training.
    Only includes images that appear in attr_map (filename keys).
    """
    manifest = []
    for root, _, files in os.walk(processed_root):
        for fn in files:
            if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
            rel = os.path.relpath(os.path.join(root, fn), processed_root)
            # attr_map keys likely are image filenames like 000001.jpg - adapt if needed
            if fn not in attr_map:
                # try without extension or with leading zeros; skip if unknown
                continue
            attr_dict = attr_map[fn]
            vec = [1 if attr_dict[a]==1 else 0 for a in required_attrs]
            manifest.append((rel, vec))
    return manifest

def train_classifier(train_manifest, processed_root, n_attrs, epochs=3, batch_size=64, lr=1e-3, device='cuda'):
    print("Training small classifier on", len(train_manifest), "samples for", epochs, "epochs")
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = ImageAttrDataset(processed_root, train_manifest, attrs=[], transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    model = AttrClassifier(n_attrs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())*imgs.size(0)
        print(f"Epoch {ep+1}/{epochs} loss: {total_loss/len(ds):.4f}")
    return model

def evaluate_mapping(mapping, forgery_root, attr_map, required_attrs, model, device='cuda', batch_size=64, threshold=0.5, out_csv=None):
    """
    For each forged image, compute attribute logits/probs and compare to target attributes.
    target attributes: if mapping records attrs, use those; otherwise look up in attr_map using real filename.
    """
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    rows = []
    model.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for rec in tqdm(mapping):
            real = rec['real']
            target_attrs = None
            if rec.get('attrs'):
                # expected to be a dict of attribute tokens -> values
                target_attrs = [int(rec['attrs'].get(a, 0)) for a in required_attrs]
            else:
                # fallback: lookup in attr_map by real filename
                bn = os.path.basename(real)
                if bn in attr_map:
                    target_attrs = [1 if attr_map[bn].get(a,0)==1 else 0 for a in required_attrs]
                else:
                    # skip if no target
                    print("Warning: no attributes found for", real)
                    continue

            for forged in rec['forgeries']:
                forged_path = os.path.join(forgery_root, os.path.basename(real).split('.')[0], forged) \
                    if os.path.isdir(os.path.join(forgery_root, os.path.basename(real).split('.')[0])) else os.path.join(forgery_root, forged)
                if not os.path.exists(forged_path):
                    # try flattened structure
                    forged_path = os.path.join(forgery_root, forged)
                if not os.path.exists(forged_path):
                    rows.append({'real':real, 'forged':forged, 'exists':False})
                    continue
                img = Image.open(forged_path).convert('RGB')
                inp = transform(img).unsqueeze(0).to(device)
                logits = model(inp)
                probs = sigmoid(logits).cpu().numpy()[0]
                # per-attribute match
                matches = [int(p >= threshold) == t for p,t in zip(probs, target_attrs)]
                # define attr_match as all targeted attributes matched (you can change to >=k)
                attr_match = all(matches)
                # confidence: mean probability across target attributes (weighted)
                if sum(target_attrs) == 0:
                    conf = float(np.mean(1-probs))  # if target all zeros, inversion
                else:
                    conf = float(np.mean([probs[i] for i,v in enumerate(target_attrs) if v==1]))
                rows.append({
                    'real': real,
                    'forged': forged,
                    'exists': True,
                    'attr_match': bool(attr_match),
                    'confidence': conf,
                    'probs': ','.join([f"{p:.3f}" for p in probs])
                })
    # write csv
    if out_csv:
        keys = ['real','forged','exists','attr_match','confidence','probs']
        with open(out_csv,'w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, '') for k in keys})
    # summary
    total = sum(1 for r in rows if r.get('exists',False))
    matched = sum(1 for r in rows if r.get('exists',False) and r.get('attr_match'))
    summary = {
        'total_checked': total,
        'matched_count': matched,
        'match_rate': matched/total if total>0 else 0.0
    }
    return rows, summary

# --------------- CLI ---------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mapping', required=True, help='mapping.json or mapping.csv linking real->forged')
    p.add_argument('--forgery-root', required=True, help='root dir where forged images are stored')
    p.add_argument('--attr-list', required=True, help='path to CelebA list_attr_celeba.txt')
    p.add_argument('--checkpoint', default=None, help='optional pretrained classifier checkpoint (.pt)')
    p.add_argument('--processed-root', default=None, help='processed images root (needed if --train)')
    p.add_argument('--train', action='store_true', help='train a small classifier if no checkpoint')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default='results/validation')
    p.add_argument('--threshold', type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    attr_map, attr_names = read_celeba_attr(args.attr_list)
    print("Loaded", len(attr_map), "attribute entries; attributes:", attr_names[:10], "...")
    mapping = load_mapping(args.mapping)
    print("Loaded mapping records:", len(mapping))

    # choose a compact set of attributes to validate (customize as needed)
    # by default we'll validate the 5 most useful attributes; user can edit this list
    validate_attrs = ['Smiling', 'Male', 'Young', 'Wearing_Hat', 'Eyeglasses']
    # ensure these exist in attr_names; if not, fallback to first 5
    validate_attrs = [a for a in validate_attrs if a in attr_names]
    if len(validate_attrs) == 0:
        validate_attrs = attr_names[:5]
    print("Validating attributes:", validate_attrs)

    n_attrs = len(attr_names)

    # load or train model
    if args.checkpoint:
        print("Loading checkpoint", args.checkpoint)
        model = AttrClassifier(len(attr_names))
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt if isinstance(ckpt, dict) and 'state_dict' not in ckpt else ckpt['state_dict'])
        model = model.to(args.device)
    elif args.train:
        if args.processed_root is None:
            raise RuntimeError("--processed-root is required when using --train")
        # build small training manifest
        train_manifest = build_manifest_for_training(args.processed_root, attr_map, validate_attrs)
        if len(train_manifest) < 100:
            raise RuntimeError("Not enough training samples found; need processed images + attr_map keys aligned")
        model = train_classifier(train_manifest, args.processed_root, n_attrs=len(validate_attrs),
                                 epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    else:
        raise RuntimeError("No checkpoint provided. Use --checkpoint or --train to obtain a classifier.")

    # Evaluate
    out_csv = os.path.join(args.out_dir, 'mapping_validated.csv')
    rows, summary = evaluate_mapping(mapping, args.forgery_root, attr_map, validate_attrs, model, device=args.device, batch_size=args.batch_size, threshold=args.threshold, out_csv=out_csv)
    summary_file = os.path.join(args.out_dir, 'summary.txt')
    with open(summary_file,'w') as f:
        f.write(json.dumps(summary, indent=2))
    print("Validation complete. Summary:", summary)
    print("Per-item results written to", out_csv)
    print("Summary written to", summary_file)

if __name__ == '__main__':
    main()
