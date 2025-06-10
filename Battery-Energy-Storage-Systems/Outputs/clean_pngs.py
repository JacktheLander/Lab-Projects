import os
import shutil

# Reference file name
reference_filename = "Threshold/ArbitrageRevenue.png"
ref_length = len(reference_filename)

# Directory setup
source_dir = "."
unused_dir = os.path.join(source_dir, "../Unused")
os.makedirs(unused_dir, exist_ok=True)

# Process PNG files
for file in os.listdir(source_dir):
    if file.endswith(".png") and len(file) > ref_length:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(unused_dir, file)
        shutil.move(src_path, dst_path)
        print(f"Moved: {file} → Unused/")
