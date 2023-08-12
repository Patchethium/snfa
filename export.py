"""
Exports PyTorch's state dict into binary file,
doesn't help in reducing model size
but makes the model readable for numpy.
"""
from typing import Dict
import torch
import argparse
import os
from hp import hp
import numpy as np


def main(args):
    assert os.path.isfile(args.model)
    if os.path.isfile(args.output):
        print(f"{args.output} exists, overwrite? [y/N]")
        if input() != str("y"):
            return
    f = open(args.output, "wb")

    # write meta data first
    # use `\0`` for splitting symbols, hope no one uses it in phone set
    phone_set_bytes = bytearray(
        "\0".join(hp["phone_set"]), "ascii"
    )  # FIXME: only ascii is supported
    meta_data = np.ascontiguousarray(
        np.array(
            [
                hp["n_fft"],
                hp["hop_size"],
                hp["win_size"],
                hp["n_mels"],
                hp["hid_dim"],
                hp["phone_dim"],
                hp["sr"],
                len(phone_set_bytes),  # phone_set_bytes_len in aligner
            ],
            dtype=np.int32,
        )
    )
    f.write(memoryview(meta_data))  # type: ignore
    f.write(memoryview(phone_set_bytes))

    state_dict: Dict[str, torch.Tensor] = torch.load(args.model, map_location="cpu")
    for k, v in state_dict.items():
        print(k, "-", v.shape)
        t = v.contiguous().view(-1).cpu().detach().type(torch.float32).numpy()
        f.write(memoryview(t))

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Exports PyTorch's state dict into binary file,
        doesn't help in reducing model size
        but makes the model readable for numpy.
        """
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="PyTorch's `.pth` state dict file",
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
