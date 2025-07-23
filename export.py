"""
Exports PyTorch's state dict into binary file,
reducing model size while making the model readable for numpy.
"""

import argparse
import os

import numpy as np
from omegaconf import OmegaConf

from train import PHONE_SET, Config, Trainer


def main(args):
    cfg = OmegaConf.load(args.config)
    cfg = Config(**cfg)

    assert os.path.isfile(args.ckpt), f"Checkpoint file {args.ckpt} does not exist."
    output = args.output + ".npz" if not args.output.endswith(".npz") else args.output
    if os.path.isfile(output):
        print(f"{output} exists, overwrite? [y/N]")
        if input() != str("y"):
            return

    data = {}
    meta_data = {
        "n_mels": cfg.n_mels,
        "sr": cfg.sr,
        "dim": cfg.dim,
        "hop_size": cfg.hop_size,
        "win_size": cfg.win_size,
        "n_fft": cfg.n_fft,
        "phone_set": "\0".join(PHONE_SET),
    }
    data["meta_data"] = meta_data
    ckpt = Trainer.load_from_checkpoint(
        args.ckpt,
        cfg=cfg,
        map_location="cpu",
    )
    aligner = ckpt.model
    aligner.eval()
    for k, v in aligner.state_dict().items():
        print(k, ":", v.shape)
        data[k] = v.detach().cpu().half().numpy()

    np.savez_compressed(output, **data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Exports PyTorch's state dict into binary file,
        doesn't help in reducing model size
        but makes the model readable for numpy.
        """
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default="config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Lightning checkpoint file to export",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="model.bin",
        help="Name of the output `.bin` file",
    )

    args = parser.parse_args()
    main(args)
