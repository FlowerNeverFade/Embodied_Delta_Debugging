#!/usr/bin/env python3
import concurrent.futures
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests


REPO_ID = "lerobot/vlabench-assets"
ENDPOINT = "https://hf-mirror.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(
    os.environ.get(
        "VLABENCH_ASSETS_DIR",
        "/root/autodl-tmp/research/VLABench/VLABench/assets",
    )
)
MANIFEST = PROJECT_ROOT / "vlabench_assets_manifest.txt"
FAILED_LIST = PROJECT_ROOT / "vlabench_assets_failed.txt"
WORKERS = int(os.environ.get("VLABENCH_ASSETS_WORKERS", "16"))


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int | None


def clear_proxy_env() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"
    os.environ["TMPDIR"] = "/root/autodl-tmp/tmp"
    os.environ["TEMP"] = "/root/autodl-tmp/tmp"
    os.environ["TMP"] = "/root/autodl-tmp/tmp"


def log(message: str) -> None:
    stamp = time.strftime("%F %T %Z")
    print(f"[{stamp}] {message}", flush=True)


def new_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "vlabench-no-proxy-assets-downloader/1.0"})
    return session


def mirror_url(url: str) -> str:
    return url.replace("https://huggingface.co", ENDPOINT)


def get_json_page(session: requests.Session, url: str) -> tuple[list[dict], str | None]:
    for attempt in range(1, 21):
        try:
            response = session.get(url, timeout=(20, 120))
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            next_url = response.links.get("next", {}).get("url")
            return response.json(), mirror_url(next_url) if next_url else None
        except Exception as exc:
            wait = min(300, 10 * attempt)
            log(f"List page attempt {attempt}/20 failed: {exc!r}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not list page after 20 attempts: {url}")


def list_files() -> list[RepoFile]:
    if MANIFEST.exists() and not os.environ.get("VLABENCH_REFRESH_MANIFEST"):
        files: list[RepoFile] = []
        with MANIFEST.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if not item["path"].endswith(".DS_Store"):
                        files.append(RepoFile(path=item["path"], size=item.get("size")))
                except json.JSONDecodeError:
                    if not line.endswith(".DS_Store"):
                        files.append(RepoFile(path=line, size=None))
        if files:
            log(f"Loaded {len(files)} files from cached manifest {MANIFEST}")
            return files

    session = new_session()
    url = (
        f"{ENDPOINT}/api/datasets/{REPO_ID}/tree/main"
        "?recursive=true&expand=false&limit=1000"
    )
    files: list[RepoFile] = []
    page = 1
    while url:
        items, url = get_json_page(session, url)
        for item in items:
            if item.get("type") == "file" and not item["path"].endswith(".DS_Store"):
                files.append(RepoFile(path=item["path"], size=item.get("size")))
        if page == 1 or page % 10 == 0:
            log(f"Listed page {page}; files so far: {len(files)}")
        page += 1

    with MANIFEST.open("w", encoding="utf-8") as handle:
        for repo_file in files:
            handle.write(json.dumps({"path": repo_file.path, "size": repo_file.size}, ensure_ascii=False) + "\n")
    return files


thread_local = threading.local()


def worker_session() -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        session = new_session()
        thread_local.session = session
    return session


def file_url(path: str) -> str:
    return f"{ENDPOINT}/datasets/{REPO_ID}/resolve/main/{quote(path, safe='/')}"


def is_complete(target: Path, expected_size: int | None) -> bool:
    if not target.exists() or not target.is_file():
        return False
    if expected_size is None:
        return target.stat().st_size > 0
    return target.stat().st_size == expected_size


def download_one(repo_file: RepoFile) -> tuple[str, str, int]:
    target = LOCAL_DIR / repo_file.path
    expected_size = repo_file.size
    if is_complete(target, expected_size):
        return ("skip", repo_file.path, expected_size or target.stat().st_size)

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial.{os.getpid()}.{threading.get_ident()}")
    url = file_url(repo_file.path)

    for attempt in range(1, 9):
        try:
            if partial.exists():
                partial.unlink()
            with worker_session().get(url, stream=True, timeout=(30, 300), allow_redirects=True) as response:
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            actual_size = partial.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise RuntimeError(f"size mismatch: got {actual_size}, expected {expected_size}")
            os.replace(partial, target)
            return ("download", repo_file.path, actual_size)
        except Exception as exc:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
            wait = min(180, 5 * attempt)
            if attempt == 8:
                return ("fail", f"{repo_file.path}\t{exc!r}", 0)
            time.sleep(wait)

    return ("fail", repo_file.path, 0)


def download_files(files: list[RepoFile]) -> list[RepoFile]:
    failed_paths: set[str] = set()
    done = skipped = downloaded = 0
    bytes_seen = 0
    total = len(files)
    last_log = time.monotonic()
    file_by_path = {repo_file.path: repo_file for repo_file in files}

    log(f"Downloading {total} files with {WORKERS} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(download_one, repo_file) for repo_file in files]
        for future in concurrent.futures.as_completed(futures):
            status, detail, size = future.result()
            done += 1
            bytes_seen += size
            if status == "skip":
                skipped += 1
            elif status == "download":
                downloaded += 1
            else:
                path = detail.split("\t", 1)[0]
                failed_paths.add(path)

            now = time.monotonic()
            if done == total or done % 100 == 0 or now - last_log >= 60:
                log(
                    "Progress: "
                    f"{done}/{total} processed, {downloaded} downloaded, "
                    f"{skipped} skipped, {len(failed_paths)} failed, "
                    f"{bytes_seen / (1024 ** 3):.2f} GiB accounted"
                )
                last_log = now

    if failed_paths:
        with FAILED_LIST.open("w", encoding="utf-8") as handle:
            for path in sorted(failed_paths):
                handle.write(path + "\n")
    else:
        FAILED_LIST.unlink(missing_ok=True)
    return [file_by_path[path] for path in sorted(failed_paths)]


def main() -> int:
    clear_proxy_env()
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    Path("/root/autodl-tmp/tmp").mkdir(parents=True, exist_ok=True)
    Path("/root/autodl-tmp/cache/huggingface").mkdir(parents=True, exist_ok=True)

    log(f"Listing {REPO_ID} from {ENDPOINT}")
    files = list_files()
    total_size = sum(repo_file.size or 0 for repo_file in files)
    log(
        f"Manifest contains {len(files)} files "
        f"({total_size / (1024 ** 3):.2f} GiB); downloading to {LOCAL_DIR}"
    )

    pending = files
    for attempt in range(1, 101):
        log(f"Download pass {attempt}/100; pending files: {len(pending)}")
        pending = download_files(pending)
        if not pending:
            break
        wait = min(600, 30 * attempt)
        log(f"{len(pending)} files failed in pass {attempt}; sleeping {wait}s before retry")
        time.sleep(wait)
    else:
        log(f"Failed after 100 attempts; remaining files: {len(pending)}")
        return 1

    existing = sum(1 for path in LOCAL_DIR.rglob("*") if path.is_file())
    downloaded_size = sum(path.stat().st_size for path in LOCAL_DIR.rglob("*") if path.is_file())
    log(f"Download finished: {existing} files, {downloaded_size / (1024 ** 3):.2f} GiB")
    if existing < max(1000, int(len(files) * 0.95)):
        log(f"Downloaded file count looks too small: {existing}/{len(files)}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
