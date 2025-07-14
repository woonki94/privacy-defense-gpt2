import os

# Find Target to merge
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "./cleaned_dataset")
output_path = os.path.join(base_dir, "./merged_dataset/merged_nano.txt")

# List txt files
txt_files = sorted([
    f for f in os.listdir(input_dir)
    if f.endswith(".txt") and f.startswith("cleaned")
])

# Merge files
with open(output_path, 'w', encoding='utf-8') as out:
    for fname in txt_files:
        fpath = os.path.join(input_dir, fname)
        print(f"Merging file: {fname}")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                out.write(line)
        out.write("\n\n")

print(f"Finish : {output_path}")
