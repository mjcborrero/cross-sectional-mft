"""HTTP transport for data.binance.vision. No domain logic lives here.

Responsibilities:
- List available object keys / prefixes via the S3-style XML listing API.
- Download files atomically (temp file + rename) with retries and backoff.
- Fetch published .CHECKSUM files and compute local SHA256 for verification.
"""

import hashlib
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .specs import DOWNLOAD_BASE, LISTING_URL

_S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"


class DownloadError(RuntimeError):
    pass


class ChecksumMismatch(RuntimeError):
    pass


class BinanceVisionClient:
    def __init__(self, retries: int = 4, backoff: float = 2.0, timeout: float = 120.0):
        self.session = requests.Session()
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout

    def _get(self, url: str, params: dict = None, stream: bool = False) -> Optional[requests.Response]:
        """GET with retries. Returns None on 404, raises DownloadError on persistent failure."""
        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, params=params, stream=stream, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 404:
                    return None
                last_err = DownloadError(f"HTTP {resp.status_code} for {url}")
            except requests.RequestException as exc:
                last_err = exc
            time.sleep(self.backoff * (2 ** attempt))
        raise DownloadError(f"giving up on {url}: {last_err}")

    def _list_page(self, prefix: str, marker: Optional[str]) -> Tuple[List[str], List[str], Optional[str]]:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        resp = self._get(LISTING_URL, params=params)
        if resp is None:
            return [], [], None
        root = ET.fromstring(resp.content)
        keys = [el.findtext(_S3 + "Key") for el in root.iter(_S3 + "Contents")]
        keys = [k for k in keys if k]
        prefixes = [el.findtext(_S3 + "Prefix") for el in root.iter(_S3 + "CommonPrefixes")]
        prefixes = [p for p in prefixes if p]
        next_marker = None
        if root.findtext(_S3 + "IsTruncated") == "true":
            next_marker = root.findtext(_S3 + "NextMarker")
            if not next_marker:
                # When NextMarker is absent, the last returned item is the marker.
                candidates = keys + prefixes
                next_marker = candidates[-1] if candidates else None
        return keys, prefixes, next_marker

    def list_keys(self, prefix: str) -> List[str]:
        """All file keys directly under prefix (paginated)."""
        out: List[str] = []
        marker = None
        while True:
            keys, _, marker = self._list_page(prefix, marker)
            out.extend(keys)
            if not marker:
                return out

    def list_prefixes(self, prefix: str) -> List[str]:
        """All sub-prefixes ('directories') directly under prefix (paginated)."""
        out: List[str] = []
        marker = None
        while True:
            _, prefixes, marker = self._list_page(prefix, marker)
            out.extend(prefixes)
            if not marker:
                return out

    def download(self, key: str, dest: Path) -> Path:
        """Atomic download: stream to .part, rename into place on success."""
        resp = self._get(DOWNLOAD_BASE + key, stream=True)
        if resp is None:
            raise DownloadError(f"404 for {key}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        return dest

    def fetch_checksum(self, key: str) -> Optional[str]:
        """Published SHA256 for a key, or None if no .CHECKSUM file exists."""
        resp = self._get(DOWNLOAD_BASE + key + ".CHECKSUM")
        if resp is None:
            return None
        text = resp.text.strip()
        return text.split()[0] if text else None

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def verify(self, key: str, path: Path) -> Optional[str]:
        """Verify a downloaded file against its published checksum.

        Returns the local SHA256 (always computed). Raises ChecksumMismatch when a
        published checksum exists and differs.
        """
        actual = self.sha256(path)
        expected = self.fetch_checksum(key)
        if expected is not None and expected.lower() != actual.lower():
            raise ChecksumMismatch(f"{key}: expected {expected}, got {actual}")
        return actual
