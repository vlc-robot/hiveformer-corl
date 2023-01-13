files_to_fix = [
    "data/processed/scannet/train_database.yaml",
    "data/processed/scannet/validation_database.yaml"
]

replacements = {
    "/projects/katefgroup/language_grounding/mask3d_processed/scannet": "/home/theophile_gervet_gmail_com/hiveformer/Mask3D/data/processed/scannet",
    "/projects/katefgroup/language_grounding/scans": "/home/theophile_gervet_gmail_com/hiveformer/Mask3D/data/raw/scannet",
}

for path in files_to_fix:
    with open(path, "r") as f:
        text = f.read()
        for k, v in replacements.items():
            text = text.replace(k, v)

    with open(path, "w") as f:
        f.write(text)
