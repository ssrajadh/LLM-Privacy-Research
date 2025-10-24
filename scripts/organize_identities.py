#!/usr/bin/env python3
"""
Usage:
 python scripts/organize_identities.py \
   --img-dir data/raw --id-file data/identities.txt --out-dir data/organized --min-per-id 5

If you don't have an id-file (image->identity), this script will infer identity from filename
if filenames encode identity (adapt as needed).
"""
import os
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm

def read_id_file(id_file):
    # expected format: `image_filename identity_id` per line (space-separated)
    mapping = {}
    with open(id_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img-dir', required=True)
    p.add_argument('--id-file', default=None)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--min-per-id', type=int, default=5)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.id_file and os.path.exists(args.id_file):
        mapping = read_id_file(args.id_file)
    else:
        # fallback: infer identity from filename prefix up to first '_' e.g., 000001_01.jpg
        mapping = {}
        for fname in os.listdir(args.img_dir):
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            # Adjust this rule to match your filenames
            id_part = fname.split('_')[0]
            mapping[fname] = id_part

    # group
    groups = defaultdict(list)
    for fname, idv in mapping.items():
        if not os.path.exists(os.path.join(args.img_dir, fname)):
            continue
        groups[idv].append(fname)

    # filter by min-per-id
    selected = {idv: files for idv, files in groups.items() if len(files) >= args.min_per_id}
    print(f"Found {len(groups)} identities; {len(selected)} with >={args.min_per_id} images")

    # copy images
    for idv, files in tqdm(selected.items()):
        outdir = os.path.join(args.out_dir, idv)
        os.makedirs(outdir, exist_ok=True)
        for fname in files:
            src = os.path.join(args.img_dir, fname)
            dst = os.path.join(outdir, fname)
            if not args.dry_run:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # save manifest
    with open(os.path.join(args.out_dir, 'manifest.csv'), 'w') as f:
        f.write('identity,num_images\n')
        for idv, files in selected.items():
            f.write(f'{idv},{len(files)}\n')

    print("Done. Organized dataset at:", args.out_dir)

if __name__ == '__main__':
    main()
