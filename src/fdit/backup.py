import os
import shutil
from pathlib import Path


def backup(archive):
    """Only explicit endpoints opt in. The archive contains no images/latents from training."""
    mounted = os.environ.get("RESULTS_BACKUP_PATH")
    repo = os.environ.get("RESULTS_BACKUP_REPO")
    if mounted:
        destination = Path(mounted)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, destination / archive.name)
    if repo:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("RESULTS_BACKUP_REPO configured but HF_TOKEN is missing")
        api = HfApi(token=token)
        info = api.repo_info(repo_id=repo, repo_type="dataset")
        if not info.private:
            raise RuntimeError("Backup requires an existing PRIVATE Hugging Face dataset repository")
        api.upload_file(path_or_fileobj=str(archive), path_in_repo=archive.name,
                        repo_id=repo, repo_type="dataset")
    if not mounted and not repo:
        print("WARNING: results may be ephemeral. Download results_bundle.tar.gz before ending the Molab session.")
