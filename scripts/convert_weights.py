import argparse
import logging

import safetensors
import torch
from safetensors.torch import save_file

from q8_kernels.functional.ops import quantize_hadamard


def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {}
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError(
                        "{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(
                            message, ckpt
                        )
                    )
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError(
                        "{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(
                            message, ckpt
                        )
                    )
            raise e
    else:
        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return (sd, metadata) if return_metadata else sd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert weights to FP8 format.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input model file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the converted model file.",
    )
    parser.add_argument(
        "--hadamard",
        action="store_true",
        help="use hadamard quantization.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    sd, metadata = load_torch_file(input_path, return_metadata=True)
    prefix = "model.diffusion_model.transformer_blocks"
    new_sd = {}
    quant_fn = lambda x, t: (x.to(t), None)
    if args.hadamard:
        quant_fn = quantize_hadamard

    for k in list(sd.keys()):
        if k.startswith(prefix):
            if "ff." in k or "attn" in k:
                if "weight" in k and sd[k].ndim == 2:
                    wfp8, _ = quant_fn(
                        sd[k].cuda().to(torch.bfloat16), torch.float8_e4m3fn
                    )
                    new_sd[k] = wfp8.cpu()
                else:
                    new_sd[k] = sd[k]
            else:
                new_sd[k] = sd[k]
        else:
            new_sd[k] = sd[k]

    save_file(new_sd, output_path, metadata=metadata)
