from pathlib import Path

raw = Path("data/raw/chest_xray")

for split in sorted(raw.iterdir()):
    if not split.is_dir():
        continue
    print(split.name)
    for label_dir in sorted(split.iterdir()):
        if not label_dir.is_dir():
            continue
        count = sum(1 for p in label_dir.iterdir() if p.is_file())
        print(f"  {label_dir.name}: {count}")cd