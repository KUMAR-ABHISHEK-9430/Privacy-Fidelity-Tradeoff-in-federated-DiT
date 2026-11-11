import os
import pytest


@pytest.mark.cuda
@pytest.mark.skipif(os.environ.get("FDIT_CUDA_TEST") != "1", reason="Requires explicit FDIT_CUDA_TEST=1 and pretrained GPU downloads")
def test_pretrained_cuda_smoke():
    from fdit.config import load_config
    from fdit.preflight import cuda_preflight, smoke
    from fdit.model import load_components
    from fdit.lora import inject
    config = load_config("configs/experiment.yaml")
    cuda_preflight()
    model, vae, scheduler, _ = load_components(config)
    model = inject(model)
    smoke(model, vae, scheduler, config)
