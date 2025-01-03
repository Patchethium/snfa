from hp import hp
from data import CommonVoiceDataset
import os
import numpy as np
from tqdm.auto import tqdm


def main():
    corpus_path = hp["corpus_path"]
    train_ds = CommonVoiceDataset(
        os.path.join(corpus_path, "train.tsv"),
        sr=hp["sr"],
        n_fft=hp["n_fft"],
        hop_size=hp["hop_size"],
        n_mels=hp["n_mels"],
    )
    dev_ds = CommonVoiceDataset(
        os.path.join(corpus_path, "dev.tsv"),
        sr=hp["sr"],
        n_fft=hp["n_fft"],
        hop_size=hp["hop_size"],
        n_mels=hp["n_mels"],
    )
    test_ds = CommonVoiceDataset(
        os.path.join(corpus_path, "test.tsv"),
        sr=hp["sr"],
        n_fft=hp["n_fft"],
        hop_size=hp["hop_size"],
        n_mels=hp["n_mels"],
    )
    out_dir = hp["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "dev"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    for name, ds in zip(["train", "dev", "test"], [train_ds, dev_ds, test_ds]):
        print(f"Processing {name} dataset")
        for i, (mel, phone) in tqdm(enumerate(ds)):
            np.save(os.path.join(out_dir, name, f"mel_{i}.npy"), mel)
            np.save(os.path.join(out_dir, name, f"phone_{i}.npy"), phone)
    print("Done!")


if __name__ == "__main__":
    main()