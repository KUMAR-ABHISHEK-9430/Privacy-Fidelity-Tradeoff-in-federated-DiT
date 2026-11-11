from types import SimpleNamespace
import torch

from fdit.diagnostic_data import balanced_public_indices, initial_noise_bank, tensor_hash, vae_reconstruction
from fdit.diffusion import generate
from test_diffusion_shapes import tiny_components


class IdentityVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=.25, latent_channels=3)
    def encode(self, x):
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=x))
    def decode(self, z):
        return SimpleNamespace(sample=z)


def test_public_validation_selection_covers_sorted_classes_without_duplicates():
    labels = [c for c in range(10) for _ in range(30)]
    selected = balanced_public_indices(list(range(300)), labels, 96, 46)
    counts = [sum(labels[i] == c for i in selected) for c in range(10)]
    assert min(counts) == 9 and max(counts) == 10
    assert len(selected) == len(set(selected)) == 96
    assert selected == balanced_public_indices(list(range(300)), labels, 96, 46)


def test_vae_scaling_roundtrip_uses_public_test_records(tmp_path):
    dataset = [(torch.rand(3, 32, 32) * 2 - 1, 0) for _ in range(8)]
    result = vae_reconstruction(dataset, [7, 6, 5, 4], IdentityVAE(), 4, 2, tmp_path, 20, lambda: None)
    assert result["mse_0_1"] == 0 and result["adequate_by_configured_psnr_heuristic"]
    assert result["ssim"] > .9999 and result["count"] == 4
    assert (tmp_path / "public_test_original_reconstruction.png").exists()


def test_paired_noise_bank_and_sampling_grouping_are_consistent():
    torch.manual_seed(9)
    model, vae, scheduler = tiny_components()
    model.eval()
    bank = initial_noise_bank(3, 4, 8, 46)
    assert tensor_hash(bank) == tensor_hash(initial_noise_bank(3, 4, 8, 46))
    torch.testing.assert_close(bank[:2], initial_noise_bank(2, 4, 8, 46), rtol=0, atol=0)
    one = torch.cat(list(generate(model, vae, scheduler, 3, 2, 46, device="cpu", batch_size=1, initial_noise=bank)))
    many = torch.cat(list(generate(model, vae, scheduler, 3, 2, 46, device="cpu", batch_size=3, initial_noise=bank)))
    torch.testing.assert_close(one, many, rtol=2e-4, atol=2e-5)
