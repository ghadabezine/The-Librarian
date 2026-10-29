"""Download the model weights used by the project from Docker Hub.

HuggingFace may be unreachable from restricted networks (it is in our build
sandbox), so we pull the same public weights from two Docker Hub sources:

* ``ai/<model>`` OCI artifacts (Docker Model Runner) -> GGUF files for the
  local LLM.
* ``semitechnologies/*`` inference images (Weaviate modules) -> the
  HuggingFace model directories baked into the image under ``app/models``.

Only the standard registry HTTP API is used; no Docker daemon is required.

Usage:
    python scripts/fetch_models.py            # everything in MODELS
    python scripts/fetch_models.py embedder   # one entry
"""
from __future__ import annotations

import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODELS = {
    # name: (kind, repository, tag, target directory or file)
    "embedder": ("image", "semitechnologies/transformers-inference",
                 "intfloat-multilingual-e5-large", "multilingual-e5-large"),
    "embedder-small": ("image", "semitechnologies/transformers-inference",
                       "sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2",
                       "paraphrase-multilingual-MiniLM-L12-v2"),
    "reranker": ("image", "semitechnologies/reranker-transformers",
                 "baai-bge-reranker-v2-m3", "bge-reranker-v2-m3"),
    "llm": ("gguf", "ai/qwen2.5", "3B-Q4_K_M", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "llm-small": ("gguf", "ai/qwen2.5", "1.5B-F16", "qwen2.5-1.5b-instruct-f16.gguf"),
}

ACCEPT = ",".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def _token(repo: str) -> str:
    url = ("https://auth.docker.io/token?service=registry.docker.io"
           f"&scope=repository:{repo}:pull")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["token"]


def _open(repo: str, path: str, accept: str | None = None):
    headers = {"Authorization": "Bearer " + _token(repo)}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/{path}", headers=headers)
    return urllib.request.urlopen(req, timeout=600)


def _manifest(repo: str, ref: str) -> dict:
    with _open(repo, f"manifests/{ref}", ACCEPT) as r:
        m = json.load(r)
    if "manifests" in m:  # multi-arch index -> pick linux/amd64
        for entry in m["manifests"]:
            p = entry.get("platform", {})
            if p.get("architecture") == "amd64" and p.get("os") == "linux":
                return _manifest(repo, entry["digest"])
        raise RuntimeError(f"no linux/amd64 manifest for {repo}:{ref}")
    return m


def _download(repo: str, digest: str, dest: Path, size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with _open(repo, f"blobs/{digest}") as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if done % (1 << 28) < (1 << 22):
                print(f"  {dest.name}: {done / 1e6:,.0f}/{size / 1e6:,.0f} MB", flush=True)
    print()
    tmp.rename(dest)


def fetch_gguf(repo: str, tag: str, target: str) -> None:
    dest = MODELS_DIR / target
    if dest.exists():
        print(f"[skip] {dest} exists")
        return
    m = _manifest(repo, tag)
    layer = next(l for l in m["layers"] if "gguf" in l["mediaType"])
    _download(repo, layer["digest"], dest, layer["size"])


def fetch_image_model(repo: str, tag: str, target: str) -> None:
    """Extract the files under ``app/models/model`` from an inference image.

    The image config history tells us which layer was created by
    ``COPY /app/models``; only that layer (~1.3 GB) is downloaded.
    """
    dest = MODELS_DIR / target
    if (dest / "config.json").exists():
        print(f"[skip] {dest} exists")
        return
    m = _manifest(repo, tag)
    with _open(repo, f"blobs/{m['config']['digest']}") as r:
        history = [h for h in json.load(r)["history"] if not h.get("empty_layer")]
    layer = next(l for h, l in zip(history, m["layers"])
                 if "/app/models" in h.get("created_by", ""))
    print(f"  model layer {layer['digest'][:19]} ({layer['size'] / 1e6:,.0f} MB)")
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    prefix = "app/models/model/"
    with _open(repo, f"blobs/{layer['digest']}") as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if not name.startswith(prefix) or not (member.isfile() or member.issym()):
                continue
            member.name = name[len(prefix):]
            tar.extract(member, tmp, filter="data")
            if member.isfile():
                print(f"    + {member.name} ({member.size / 1e6:,.1f} MB)")
    # Some images store a HuggingFace cache (models--org--name/snapshots/<rev>/
    # with symlinks into blobs/); others store a plain model directory.
    configs = [c for c in tmp.rglob("config.json") if ".no_exist" not in c.parts]
    if not configs:
        raise RuntimeError(f"no model found in {repo}:{tag}")
    src_dir = min(configs, key=lambda c: len(c.parts)).parent
    if src_dir == tmp:
        tmp.rename(dest)
    else:
        shutil.copytree(src_dir, dest, symlinks=False)
        shutil.rmtree(tmp)


def main(names: list[str]) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for name in names or list(MODELS):
        kind, repo, tag, target = MODELS[name]
        print(f"[{name}] {repo}:{tag}")
        (fetch_gguf if kind == "gguf" else fetch_image_model)(repo, tag, target)


if __name__ == "__main__":
    main(sys.argv[1:])
