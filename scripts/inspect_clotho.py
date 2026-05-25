from datasets import Audio, load_dataset

for split in ("train", "validation", "test"):
    print(f"\n========== split={split} ==========")
    ds = load_dataset("CLAPv2/Clotho", split=split)

    # Don't decode audio — we only care about text here.
    ds = ds.cast_column("audio", Audio(decode=False))

    print("Features:")
    for name, feat in ds.features.items():
        print(f"  {name}: {feat}")

    print(f"\nColumn names: {ds.column_names}")
    print(f"Num rows: {len(ds)}")

    for i in range(3):
        row = ds[i]
        print(f"\n--- {split}[{i}] ---")
        print(f"  index:       {row['index']}")
        print(f"  datasetname: {row['datasetname']}")
        print(f"  audio_len:   {row['audio_len']}")
        print(f"  text (len={len(row['text'])}): {repr(row['text'])[:400]}")
        rt = row["raw_text"]
        print(f"  raw_text: type={type(rt).__name__}, len={len(rt)}")
        for j, cap in enumerate(rt):
            print(f"    [{j}] (len={len(cap)}): {repr(cap)}")
