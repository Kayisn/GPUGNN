import requests
import argparse
from pathlib import Path

def download_snap_graph(url, filename):
    """
    Downloads and a SNAP graph file. E.g.: https://snap.stanford.edu/data/wiki-Vote.txt.gz
    """
    if not filename.exists():
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.raw.read())
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download {url}")
    else:
        print(f"File already exists: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Download a SNAP network and save it to the graphs/snap folder.")
    parser.add_argument("snap_url", type=str, help="URL of the SNAP network")
    args = parser.parse_args()

    snap_url = args.snap_url
    snap_filename = Path("graphs") / "snap" / Path(snap_url).name

    # Ensure the directory exists
    snap_filename.parent.mkdir(parents=True, exist_ok=True)

    # Download the graph
    download_snap_graph(snap_url, snap_filename)

if __name__ == "__main__":
    main()
