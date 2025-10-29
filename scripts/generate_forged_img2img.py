# scripts/generate_forged_img2img.py
import os, csv, json, random, argparse
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms

def load_metadata(attr_file_path, identity_file_path=None):
    """
    Load CelebA attributes from list_attr_celeba.txt
    Optionally load identity mapping from list_identity_celeba.txt for organized directories
    Returns dict: filename -> {attribute_name: value, identity_id: id}
    
    Format of list_attr_celeba.txt:
    Line 1: number of images
    Line 2: space-separated attribute names
    Line 3+: filename.jpg followed by -1/1 values for each attribute
    
    Format of list_identity_celeba.txt:
    filename.jpg identity_id
    """
    metadata = {}
    
    if not os.path.exists(attr_file_path):
        print(f"Warning: Attribute file not found at {attr_file_path}")
        return metadata
    
    # Load identity mapping if provided
    identity_map = {}
    if identity_file_path and os.path.exists(identity_file_path):
        with open(identity_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    identity_map[parts[0]] = parts[1]
        print(f"Loaded identity mapping for {len(identity_map)} images")
    
    with open(attr_file_path, 'r') as f:
        lines = f.readlines()
        
        # First line is count, second line is attribute names
        num_images = int(lines[0].strip())
        attr_names = lines[1].strip().split()
        
        # Process each image
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            filename = parts[0]
            # Parse attributes (-1 means absent, 1 means present)
            attrs = {}
            for i, attr_name in enumerate(attr_names):
                if i + 1 < len(parts):
                    value = int(parts[i + 1])
                    # Convert -1/1 to True/False or descriptive strings
                    if value == 1:
                        attrs[attr_name] = attr_name.replace('_', ' ')
                    else:
                        # For negatives, we can use "not X" or just omit
                        attrs[attr_name] = ""
            
            # Add identity if available
            if filename in identity_map:
                attrs['identity_id'] = identity_map[filename]
            
            metadata[filename] = attrs
    
    print(f"Loaded metadata for {len(metadata)} images")
    return metadata

def build_prompt(attrs, template):
    """
    Fill attribute tokens into template
    Safely handles missing keys by replacing them with empty strings
    """
    # Create a safe dict that returns empty string for missing keys
    safe_attrs = {k: v for k, v in attrs.items()}
    
    # Try to format the template, if keys are missing just use the template as-is
    try:
        return template.format(**safe_attrs)
    except KeyError as e:
        # If template has placeholders not in attrs, use a default prompt
        print(f"Warning: Template key {e} not found in attributes, using default prompt")
        return f"a {safe_attrs.get('quality', 'high-quality')} {safe_attrs.get('style', 'portrait photo')}"

def generate_forgeries(pipe, real_img_path, prompt, out_dir, n, seeds, denoise, device, cfg_scale=7.5, steps=25):
    """
    Generate forgeries using img2img diffusion
    
    Args:
        pipe: Stable Diffusion pipeline
        real_img_path: Path to real image
        prompt: Text prompt for generation
        out_dir: Output directory
        n: Number of forgeries to generate
        seeds: List of seeds for each forgery
        denoise: Denoising strength (0-1). Lower preserves more identity, higher changes more
        device: Device to use (cuda/cpu)
        cfg_scale: Classifier-free guidance scale
        steps: Number of inference steps (20-30 recommended for faster generation)
    
    Returns:
        List of dicts with filename and seed for each generated image
    """
    img = Image.open(real_img_path).convert("RGB")
    results = []
    
    for i in range(n):
        seed = seeds[i]
        generator = torch.Generator(device=device).manual_seed(seed)
        
        output = pipe(
            prompt=prompt, 
            image=img, 
            strength=denoise,
            guidance_scale=cfg_scale, 
            num_inference_steps=steps, 
            generator=generator
        )
        
        forged = output.images[0]
        fname = f"{os.path.splitext(os.path.basename(real_img_path))[0]}_forg_{seed}.png"
        forged.save(os.path.join(out_dir, fname))
        results.append({"filename": fname, "seed": int(seed)})
        
        print(f"  Generated {i+1}/{n}: {fname}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forged images using Stable Diffusion img2img")
    parser.add_argument("--model", required=True, help="Stable Diffusion model name or path")
    parser.add_argument("--in-dir", required=True, help="Input directory with real images")
    parser.add_argument("--meta", required=True, help="Path to list_attr_celeba.txt metadata file")
    parser.add_argument("--identity-meta", default=None, help="Path to list_identity_celeba.txt (optional, for organized dirs)")
    parser.add_argument("--out-dir", required=True, help="Output directory for forged images")
    parser.add_argument("--d", type=int, default=3, help="Number of forgeries per image")
    parser.add_argument("--template", default="a high-quality portrait photo of a person", 
                        help="Prompt template for generation")
    parser.add_argument("--denoise", type=float, default=0.45, 
                        help="Denoising strength (0-1). Lower=preserve identity, Higher=change more")
    parser.add_argument("--cfg-scale", type=float, default=7.5, 
                        help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=25, 
                        help="Number of inference steps (20-30 recommended for speed)")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit number of images to process (for testing)")
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model, 
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        safety_checker=None,  # Disable NSFW filter for research purposes
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    print("Model loaded successfully")

    # Load metadata
    print(f"Loading metadata from: {args.meta}")
    meta = load_metadata(args.meta, args.identity_meta)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Process images and generate forgeries
    mapping_rows = []
    processed = 0
    
    for idx, (fname, attrs) in enumerate(meta.items()):
        # Check limit
        if args.limit and processed >= args.limit:
            print(f"\nReached limit of {args.limit} images, stopping.")
            break
            
        # Try to find the image in input directory
        # First try direct path (for raw directory)
        real_path = os.path.join(args.in_dir, fname)
        
        # If not found, try with .png extension
        if not os.path.exists(real_path):
            alt = fname.replace('.jpg', '.png')
            real_path = os.path.join(args.in_dir, alt)
        
        # If still not found, try in identity subdirectory (for organized directory)
        if not os.path.exists(real_path):
            identity_id = attrs.get('identity_id')
            if identity_id:
                real_path = os.path.join(args.in_dir, identity_id, fname)
                if not os.path.exists(real_path):
                    alt = fname.replace('.jpg', '.png')
                    real_path = os.path.join(args.in_dir, identity_id, alt)
        
        # Skip if still not found
        if not os.path.exists(real_path):
            if idx < 5:  # Only print first few warnings
                print(f"  Skipping {fname} - file not found")
            continue

            
        print(f"\n[{processed+1}] Processing: {fname}")
        
        # Build prompt
        prompt = build_prompt(attrs, args.template)
        print(f"  Prompt: {prompt}")
        
        # Generate random seeds for reproducibility
        seeds = [random.randint(0, 2**30) for _ in range(args.d)]
        
        # Generate forgeries
        forged_info = generate_forgeries(
            pipe, real_path, prompt, args.out_dir, 
            args.d, seeds, args.denoise, device,
            cfg_scale=args.cfg_scale, steps=args.steps
        )
        
        # Record mapping with all metadata
        mapping_row = {
            "real": fname,
            "forged": [r["filename"] for r in forged_info],
            "prompt": prompt,
            "seeds": [r["seed"] for r in forged_info],
            "model": args.model,
            "cfg_scale": args.cfg_scale,
            "steps": args.steps,
            "denoising_strength": args.denoise
        }
        mapping_rows.append(mapping_row)
        processed += 1

    # Save mapping as JSON
    mapping_json_path = os.path.join(args.out_dir, "mapping.json")
    with open(mapping_json_path, "w") as f:
        json.dump(mapping_rows, f, indent=2)
    print(f"\nSaved JSON mapping to: {mapping_json_path}")
    
    # Also save as CSV for easier viewing
    mapping_csv_path = os.path.join(args.out_dir, "mapping.csv")
    with open(mapping_csv_path, "w", newline='') as f:
        if mapping_rows:
            # Flatten forged list and seeds list for CSV
            fieldnames = ["real", "prompt", "model", "cfg_scale", "steps", 
                         "denoising_strength"]
            # Add columns for each forgery
            max_forgeries = max(len(row["forged"]) for row in mapping_rows)
            for i in range(max_forgeries):
                fieldnames.extend([f"forged{i+1}", f"seed{i+1}"])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in mapping_rows:
                csv_row = {
                    "real": row["real"],
                    "prompt": row["prompt"],
                    "model": row["model"],
                    "cfg_scale": row["cfg_scale"],
                    "steps": row["steps"],
                    "denoising_strength": row["denoising_strength"]
                }
                # Add forged images and seeds
                for i, (forged, seed) in enumerate(zip(row["forged"], row["seeds"])):
                    csv_row[f"forged{i+1}"] = forged
                    csv_row[f"seed{i+1}"] = seed
                
                writer.writerow(csv_row)
    
    print(f"Saved CSV mapping to: {mapping_csv_path}")
    print(f"\nProcessed {processed} images, generated {processed * args.d} forgeries")
    print(f"All forgeries saved to: {args.out_dir}")
