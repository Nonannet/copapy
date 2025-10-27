import os
import requests

OWNER = "Nonannet"
REPO = "copapy"

def main() -> None:
    # Get all releases (includes prereleases)
    api_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases"
    resp = requests.get(api_url, timeout=10)
    resp.raise_for_status()
    releases = resp.json()

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
        else:
            dest = ''

        if dest:
            os.makedirs(dest, exist_ok=True)
            print(f"- Downloading {name} ...")
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(os.path.join(dest, name), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"- Saved {name} to {dest}")

if __name__ == "__main__":
    main()
