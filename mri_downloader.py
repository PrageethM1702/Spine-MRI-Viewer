import os
import subprocess
import shutil

DATASET_REPO_URL = "https://github.com/spine-generic/data-multi-subject"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
REPO_DIR = os.path.join(BASE_DIR, "data", "data-multi-subject")

_last_error = ""


def get_last_error() -> str:
    return _last_error


def _set_error(message: str) -> None:
    global _last_error
    _last_error = (message or "").strip()


def get_cached_path(subject: str) -> str:
    return os.path.join(CACHE_DIR, subject, "anat", f"{subject}_T2w.nii.gz")


def is_cached(subject: str) -> bool:
    path = get_cached_path(subject)
    return os.path.exists(path) and os.path.getsize(path) > 1024


def _run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        _set_error(f"'{cmd[0]}' is not installed or not on PATH")
        return None


def _ensure_repo_cloned() -> bool:
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        return True

    os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)

    staging = REPO_DIR + ".cloning"
    if os.path.exists(staging):
        shutil.rmtree(staging, ignore_errors=True)

    # Do NOT use --depth=1: shallow clones break git-annex because the
    # annex special remote needs to resolve object hashes that may not
    # exist in a truncated history.
    result = _run(["git", "clone", DATASET_REPO_URL, staging])
    if result is None:
        return False
    if result.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        _set_error(f"Could not clone the dataset repository.\n{result.stderr}")
        return False

    os.makedirs(REPO_DIR, exist_ok=True)
    for entry in os.listdir(staging):
        destination = os.path.join(REPO_DIR, entry)
        if os.path.exists(destination):
            continue
        shutil.move(os.path.join(staging, entry), destination)
    shutil.rmtree(staging, ignore_errors=True)

    # Initialise git-annex and enable the ComputeCanada public remote,
    # which is the current hosting provider for this dataset (changed 2025).
    for cmd in (
        ["git", "annex", "init"],
        ["git", "annex", "enableremote", "computecanada-public"],
    ):
        r = _run(cmd, cwd=REPO_DIR)
        if r is None:
            return False
        if r.returncode != 0:
            # enableremote failure is non-fatal if the remote is already
            # configured; continue rather than aborting.
            if cmd[2] != "enableremote":
                _set_error(f"{' '.join(cmd)} failed.\n{r.stderr}")
                return False

    return True


def download_subject(subject: str, progress_callback=None) -> str | None:
    _set_error("")
    cached_path = get_cached_path(subject)

    if is_cached(subject):
        return cached_path

    if not _ensure_repo_cloned():
        return None

    annex_path = os.path.join(subject, "anat", f"{subject}_T2w.nii.gz")

    if progress_callback:
        progress_callback(0, 0, 0)

    result = _run(["git", "annex", "get", annex_path], cwd=REPO_DIR)
    if result is None:
        return None
    if result.returncode != 0:
        _set_error(f"Could not download {subject}.\n{result.stderr}")
        return None

    annex_full_path = os.path.join(REPO_DIR, annex_path)
    if not os.path.exists(annex_full_path) or os.path.getsize(annex_full_path) < 1024:
        _set_error(
            f"{subject} was reported as downloaded but no data arrived. "
            "The dataset remote may be unreachable."
        )
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
