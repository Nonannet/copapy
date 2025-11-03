from typing import Any
import os
import json
from urllib.request import urlopen, Request

OWNER = "Nonannet"
REPO = "copapy"

def fetch_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "Python"})
    with urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        return json.load(resp)

def download_file(url: str, dest_path: str) -> None:
    req = Request(url, headers={"User-Agent": "Python"})
    with urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())

def main() -> None:
    # Get all releases (includes prereleases)
    api_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases"
    releases = fetch_json(api_url)

    assert releases, "No releases found."

    release = releases[0]  # newest release (first in list)
    tag = release["tag_name"]
    print(f"Found latest release: {tag}")

    assets = release.get("assets", [])
    assert assets, "No assets found for this release."

    for asset in assets:
        url = asset["browser_download_url"]
        name: str = asset["name"]

        if name.endswith('.o'):
            dest = 'src/copapy/obj'
        elif name == 'coparun.exe' and os.name == 'nt':
            dest = 'bin'
        elif name == 'coparun' and os.name == 'posix':
            dest = 'bin'
        elif name.startswith('coparun-'):
            dest = 'bin'
        else:
            dest = ''

        if dest:
            os.makedirs(dest, exist_ok=True)
            print(f"- Downloading {name} ...")
            download_file(url, os.path.join(dest, name))
            print(f"- Saved {name} to {dest}")

if __name__ == "__main__":
    main()
