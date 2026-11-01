"""
Exports PyTorch's state dict into binary file,
reducing model size while making the model readable for numpy and Rust.
"""

import argparse
import json
import os

import numpy as np

FORMATS = ("npz", "safetensors")


def from_checkpoint(config, ckpt):
    """Loads half-precision weights and metadata from a Lightning checkpoint."""
    from omegaconf import OmegaConf

    from train import PHONE_SET, Config, Trainer

    cfg = OmegaConf.load(config)
    cfg = Config(**cfg)

    assert os.path.isfile(ckpt), f"Checkpoint file {ckpt} does not exist."
    meta_data = {
        "n_mels": cfg.n_mels,
        "sr": cfg.sr,
        "dim": cfg.dim,
        "hop_size": cfg.hop_size,
        "win_size": cfg.win_size,
        "n_fft": cfg.n_fft,
        "phone_set": "\0".join(PHONE_SET),
    }
    ckpt = Trainer.load_from_checkpoint(
        ckpt,
        cfg=cfg,
        map_location="cpu",
    )
    aligner = ckpt.model
    aligner.eval()
    encoder = aligner.encoder
    weights = {}
    for k, v in encoder.state_dict().items():
        print(k, ":", v.shape)
        weights[k] = v.detach().cpu().half().numpy()
    return weights, meta_data


def from_npz(npz):
    """Loads weights and metadata from a previously exported `.npz` file."""
    assert os.path.isfile(npz), f"Weight file {npz} does not exist."
    data = np.load(npz, allow_pickle=True)
    meta_data = data["meta_data"].item()
    weights = {k: v for k, v in data.items() if k != "meta_data"}
    return weights, meta_data


def save_npz(output, weights, meta_data):
    np.savez_compressed(output, meta_data=meta_data, **weights)


def save_safetensors(output, weights, meta_data):
    """
    Saves the weights with the metadata in the header, each value JSON-encoded,
    since safetensors metadata maps strings to strings.
    """
    from safetensors.numpy import save_file

    metadata = {k: json.dumps(v) for k, v in meta_data.items()}
    save_file(weights, output, metadata=metadata)


def main(args):
    if (args.ckpt is None) == (args.npz is None):
        raise SystemExit("Pass exactly one of --ckpt and --npz.")

    ext = "." + args.format
    output = args.output if args.output.endswith(ext) else args.output + ext
    if os.path.isfile(output):
        print(f"{output} exists, overwrite? [y/N]")
        if input() != str("y"):
            return

    if args.ckpt is not None:
        weights, meta_data = from_checkpoint(args.config, args.ckpt)
    else:
        weights, meta_data = from_npz(args.npz)

    if args.format == "npz":
        save_npz(output, weights, meta_data)
    else:
        save_safetensors(output, weights, meta_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Exports PyTorch's state dict into binary file,
        doesn't help in reducing model size
        but makes the model readable for numpy and Rust.
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
        required=False,
        help="Lightning checkpoint file to export",
    )
    parser.add_argument(
        "--npz",
        type=str,
        required=False,
        help="Exported `.npz` file to convert, instead of a checkpoint",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=FORMATS,
        default="npz",
        help="Output format: `npz` for numpy, `safetensors` for the Rust crate",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="model",
        help="Name of the output file, the extension is added if missing",
    )

    args = parser.parse_args()
    main(args)
