"""Copy the supplied course data after verifying every expected file hash."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;DEST=ROOT/'data'
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);args=p.parse_args()
    source=args.source.resolve();manifest=json.loads((DEST/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        src=source/name
        if not src.is_file():raise FileNotFoundError('Missing course file: '+str(src))
        if hashlib.sha256(src.read_bytes()).hexdigest()!=digest:raise ValueError('Hash mismatch: '+name)
    for name in manifest['sha256']:
        src=source/name;target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
        if src.resolve()!=target.resolve():shutil.copy2(src,target)
    print('Course data installed and all manifest hashes verified.')
if __name__=='__main__':main()
