"""Verify source/checkpoint identity; optional complete release verification."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');args=p.parse_args()
    m=json.loads((ROOT/'MODEL_MANIFEST.json').read_text())
    for name,digest in {**m['source_sha256'],**m['assets_sha256']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    actual=sum((ROOT/name).stat().st_size for name in m['assets_sha256'])
    assert actual==m['asset_bytes'] and actual<16*2**20
    if args.all:
        for name,digest in json.loads((ROOT/'FILE_MANIFEST.json').read_text())['sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print(json.dumps(dict(status='passed',version=m['version'],asset_bytes=actual)))
if __name__=='__main__':main()
