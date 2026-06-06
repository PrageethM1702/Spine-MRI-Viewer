import os
import subprocess
import shutil

DATASET_REPO_URL = "https://github.com/spine-generic/data-multi-subject"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
REPO_DIR = os.path.join(BASE_DIR, "data", "data-multi-subject")


def get_cached_path(subject: str) -> str:
    return os.path.join(CACHE_DIR, subject, "anat", f"{subject}_T2w.nii.gz")


def is_cached(subject: str) -> bool:
    path = get_cached_path(subject)
    return os.path.exists(path) and os.path.getsize(path) > 1024


def _ensure_repo_cloned() -> bool:
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        return True
    os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)

    # Do NOT use --depth=1: shallow clones break git-annex because the
    # annex special remote needs to resolve object hashes that may not
    # exist in a truncated history.
    result = subprocess.run(
        ["git", "clone", DATASET_REPO_URL, REPO_DIR],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"git clone failed: {result.stderr}")
        return False

    # Initialise git-annex and enable the ComputeCanada public remote,
    # which is the current hosting provider for this dataset (changed 2025).
    for cmd in (
        ["git", "annex", "init"],
        ["git", "annex", "enableremote", "computecanada-public"],
    ):
        r = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{' '.join(cmd)} failed: {r.stderr}")
            # enableremote failure is non-fatal if the remote is already
            # configured; continue rather than aborting.
            if cmd[1] != "enableremote":
                return False

    return True


def download_subject(subject: str, progress_callback=None) -> str | None:
    cached_path = get_cached_path(subject)

    if is_cached(subject):
        return cached_path

    if not _ensure_repo_cloned():
        print(f"Repository not available for {subject}")
        return None

    annex_path = os.path.join(subject, "anat", f"{subject}_T2w.nii.gz")

    if progress_callback:
        progress_callback(0, 0, 0)

    result = subprocess.run(
        ["git", "annex", "get", annex_path],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"git annex get failed for {subject}: {result.stderr}")
        return None

    annex_full_path = os.path.join(REPO_DIR, annex_path)
    if not os.path.exists(annex_full_path) or os.path.getsize(annex_full_path) < 1024:
        print(f"File missing or too small after annex get: {annex_full_path}")
        return None

    os.makedirs(os.path.dirname(cached_path), exist_ok=True)
    shutil.copy2(annex_full_path, cached_path)

    if progress_callback:
        size = os.path.getsize(cached_path)
        progress_callback(size, size, 100)

    return cached_path


def clear_cache(subject: str = None):
    if subject:
        path = os.path.join(CACHE_DIR, subject)
        if os.path.exists(path):
            shutil.rmtree(path)
    else:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
