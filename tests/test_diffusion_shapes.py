from types import SimpleNamespace

import torch
import pytest
from diffusers import DiTTransformer2DModel, DDIMScheduler, AutoencoderKL

from fdit.diffusion import epsilon_channels, loss_vector
from fdit.model import verify_components


def tiny_components():
    model = DiTTransformer2DModel(num_attention_heads=2, attention_head_dim=8,
                                  in_channels=4, out_channels=8, num_layers=1,
                                  sample_size=8, patch_size=2, num_embeds_ada_norm=1000,
                                  norm_type="ada_norm_zero")
    vae = AutoencoderKL(block_out_channels=(8,), norm_num_groups=4, latent_channels=4)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", clip_sample=False)
    return model, vae, scheduler


def test_null_class_channels_and_sampling_step():
    model, vae, scheduler = tiny_components()
    info = verify_components(model, vae, scheduler)
    assert info["null_class_id"] == 1000
    assert info["vae_scaling_factor"] > 0
    x = torch.randn(2, 4, 8, 8)
    vector = loss_vector(model, scheduler, x)
    assert vector.shape == (2,) and torch.isfinite(vector).all()
    scheduler.set_timesteps(2)
    t = scheduler.timesteps[0]
    output = model(x, timestep=t.expand(2), class_labels=torch.full((2,), 1000)).sample
    assert output.shape == (2, 8, 8, 8)
    eps = epsilon_channels(output, x)
    assert torch.equal(eps, output[:, :4])
    assert scheduler.step(eps, t, x).prev_sample.shape == x.shape
    with pytest.raises(ValueError):
        epsilon_channels(output[:, :4], x)


def test_incompatible_null_fails():
    model, vae, scheduler = tiny_components()
    model.transformer_blocks[0].norm1.emb.class_embedder.embedding_table = torch.nn.Embedding(1000, 16)
    with pytest.raises(RuntimeError, match="null ID"):
        verify_components(model, vae, scheduler)
