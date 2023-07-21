import torch
from vap.modules.lightning_module import VAPModule
from vap.utils.utils import everything_deterministic

everything_deterministic()


def test_causality_gradient(
    model,
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
    x = torch.randn(4, 2, n_samples, device=model.device, requires_grad=True)

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

    model = VAPModule.load_model("example/checkpoints/checkpoint.ckpt").to("cpu")

    test_causality_gradient(model, duration=10, focus_time=5, verbose=True)

    model.to("cuda")
    test_causality_gradient(model, duration=10, focus_time=5, verbose=True)
