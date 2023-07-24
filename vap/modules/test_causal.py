import torch
from vap.modules.lightning_module import VAPModule, VAP
from vap.utils.utils import everything_deterministic

everything_deterministic()


def test_causality_gradient(
    model: VAP,
    duration: float = 10.0,
    focus_time: float = 5.0,
    pad_frames: int = 0,
    verbose: bool = False,
) -> bool:
    """
    Test that the gradient is zero in the future.
    1. Create a random waveform of duration `duration` and set `requires_grad=True`
    2. Extract the model output (logits)
    3. Choose a frame in the middle of the sequence
    4. Calculate the loss gradient, on that specific frame, w.r.t the input tensor
    5. There should not be any gradient information in the future part of the input tensor
    """
    model.train()

    # Generate a random waveform
    n_samples = int(model.sample_rate * duration)
    focus_sample = int(model.sample_rate * focus_time)
    focus_frame = int(model.frame_hz * focus_time)
    # 1. Waveform + gradient tracking
    x = torch.randn(2, 2, n_samples, device=model.device, requires_grad=True)

    # 2. Model output
    out = model(x)

    # 3. Gradient calculation
    loss = out["logits"][:, focus_frame, :].sum()
    loss.backward()

    # Gradient result
    g = x.grad.abs()
    future_grad = g[:, focus_sample + pad_frames :].sum().item()
    past_grad = g[:, :focus_sample].sum().item()
    is_causal = future_grad == 0
    if verbose:
        print(f"({model.device}) Future total gradient:", future_grad)
        print(f"({model.device}) Past total gradient:", past_grad)
        if not is_causal:
            print(f"Future gradient should be zero. got  {future_grad}")

    return is_causal


if __name__ == "__main__":

    from argparse import ArgumentParser
    from vap.modules.modules import TransformerStereo
    from vap.modules.encoder import EncoderCPC
    from vap.modules.encoder_hubert import EncoderHubert

    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, default="cpc")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--focus", type=float, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is not None:
        model = VAPModule.load_model(args.checkpoint)

    else:
        if args.encoder.lower() == "hubert":
            enc = EncoderHubert()
        else:
            enc = EncoderCPC()
        model = VAP(enc, TransformerStereo())

    if args.cpu:
        model = model.to("cpu")
        print("CPU")
    else:
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("CUDA")

    test_causality_gradient(model, duration=10, focus_time=5, verbose=True)
