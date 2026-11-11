import math

import torch


def verify_components(model, vae, scheduler):
    c = model.config
    if c.in_channels != 4 or c.out_channels != 8 or c.num_embeds_ada_norm != 1000:
        raise RuntimeError("Unsupported DiT channel/class configuration")
    embeddings = [m for n, m in model.named_modules()
                  if n.endswith("class_embedder.embedding_table") and isinstance(m, torch.nn.Embedding)]
    if not embeddings or any(m.num_embeddings != 1001 for m in embeddings):
        raise RuntimeError("Cannot verify pretrained unconditional/null ID 1000")
    scale = float(vae.config.scaling_factor)
    if not math.isfinite(scale) or scale <= 0 or vae.config.latent_channels != c.in_channels:
        raise RuntimeError("Incompatible VAE scaling/channels")
    sc = scheduler.config
    if sc.prediction_type != "epsilon" or sc.num_train_timesteps != 1000:
        raise RuntimeError("Unsupported pretrained scheduler")
    return {"null_class_id": 1000, "vae_scaling_factor": scale,
            "vae_scaling_uses_library_default": "scaling_factor" in getattr(vae.config, "_use_default_values", []),
            "scheduler": dict(sc), "transformer": dict(c), "vae": dict(vae.config)}


def load_components(config):
    from diffusers import DiTTransformer2DModel, AutoencoderKL, DDIMScheduler
    from huggingface_hub import model_info
    info = model_info(config["model_id"], revision=config["model_revision"])
    revision = info.sha
    kwargs = {"revision": revision}
    model = DiTTransformer2DModel.from_pretrained(config["model_id"], subfolder="transformer", **kwargs)
    vae = AutoencoderKL.from_pretrained(config["model_id"], subfolder="vae", **kwargs)
    scheduler = DDIMScheduler.from_pretrained(config["model_id"], subfolder="scheduler", **kwargs)
    model.requires_grad_(False)
    vae.requires_grad_(False).eval()
    metadata = verify_components(model, vae, scheduler)
    metadata.update(model_revision=revision, model_card="https://huggingface.co/" + config["model_id"],
                    original_model_card=info.card_data.to_dict() if info.card_data else {},
                    denoiser_pretraining="ImageNet-1K (not EuroSAT)",
                    model_license="CC BY-NC 4.0", vae_id=config["model_id"] + "/vae")
    return model.to("cuda"), vae.to("cuda"), scheduler, metadata
