from argparse import ArgumentParser
import torch
from vap.modules.VAP import VAP
from vap.modules.lightning_module import VAPModule


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--state_dict", type=str, default="vap_state_dict.pt")
    parser.add_argument("--module_state_dict", action="store_true")
    args = parser.parse_args()

    mod_type = "Model"
    if args.module_state_dict:
        model = VAPModule.load_from_checkpoint(args.checkpoint)
        mod_type = "LightningModule"
    else:
        model = VAPModule.load_model(args.checkpoint)
    sd = model.state_dict()
    torch.save(sd, args.state_dict)
    print(f"Saved VAP ({mod_type}) state dict to {args.state_dict}")
