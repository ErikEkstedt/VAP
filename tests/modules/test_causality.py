import torch
import pytest

from vap.modules.VAP import VAP
from vap.modules.encoder import EncoderCPC
from vap.modules.encoder_mms import EncoderMMS
from vap.modules.modules import TransformerStereo

# from vap.modules.causal_testing import causal_test_samples_to_frames
from vap.modules.test_causal import causal_test_samples_to_frames


def test_encoder_cpc_causality():
    """CPC is causal with pad_samples >= 311"""
    enc = EncoderCPC()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(device)
    is_causal, *_ = causal_test_samples_to_frames(
        enc, 10, 5, pad_samples=311, wav_channels=1, device=device
    )
    assert is_causal, "CPC is not causal!"


@pytest.mark.parametrize("use_feature_projection", [True, False])
def test_encoder_mms_causality(use_feature_projection):
    enc = EncoderMMS(use_feature_projection=use_feature_projection)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(device)
    is_causal, *_ = causal_test_samples_to_frames(
        enc, 10, 5, pad_samples=399, wav_channels=1, device=device
    )
    assert is_causal, "MMS is not causal!"


@pytest.mark.parametrize(
    "encoder_name,dim",
    [
        ("cpc", 256),  # Standard
        ("cpc", 128),  # Non-standard
        ("mms", 512),  # Standard
        ("mms", 256),  # Non-Standard
    ],
)
def test_vap_causality(encoder_name, dim):
    """MMS is causal with pad_samples >= 399"""
    if encoder_name == "cpc":
        enc = EncoderCPC()
        pad_samples = 311
    else:
        enc = EncoderMMS()
        pad_samples = 399

    model = VAP(enc, TransformerStereo(dim=dim))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    is_causal, *_ = causal_test_samples_to_frames(
        model, 10, 5, pad_samples=pad_samples, wav_channels=2, device=device
    )
    assert is_causal, f"VAP {encoder_name} -> {dim} is not causal!"
