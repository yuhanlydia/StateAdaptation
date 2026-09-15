"""One-time, checksum-verified completion of the interrupted paper migration."""
from pathlib import Path, PurePosixPath
import base64, hashlib, json, lzma, urllib.request

parts = [Path('migration_payload') / f'paper_part{i:02d}.b64' for i in range(5)]
if not all(p.is_file() for p in parts):
    if Path('paper/visual_lens/main.tex').is_file():
        print('Manuscript already installed; no payload reapplied.')
        raise SystemExit(0)
    raise SystemExit('Incomplete manuscript migration payload')
raw = bytearray(base64.b64decode(''.join(p.read_text().strip() for p in parts), validate=True))
# The reviewed continuation includes the complete combined prompt file. This
# changes the LZMA2 first-chunk sizes; the prior four staged chunks are unchanged.
if raw[24] != 0xe2 or raw[25:29] != bytes.fromhex('199c9715'):
    raise SystemExit('Unexpected staged payload header')
raw[25:29] = bytes.fromhex('d1c99594')
expected = '764b4183f7f4de3e97028ecc8795a98e7680792c21be811772af2f9fbdd04b43'
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit('Manuscript payload checksum mismatch')
files = json.loads(lzma.decompress(raw))
if len(files) != 18:
    raise SystemExit('Unexpected manuscript source file count')
for name, text in files.items():
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.parts[:2] != ('paper', 'visual_lens'):
        raise SystemExit(f'Unsafe manuscript path: {name}')
    out = Path(name)
    content = text.encode('utf-8')
    if out.exists() and out.read_bytes() != content:
        raise SystemExit(f'Refusing to overwrite changed manuscript source: {name}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(content)

styles = {
 'fancyhdr.sty': ('77ed4e3012d822c7cca5c17efcae308b32b8cc2b','b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea'),
 'natbib.sty': ('ff0d0b91b6ef41468c593a0ca40a81f9a183b055','88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d'),
 'iclr2027_conference.sty': ('f61ad7efce0855557694078c0945e6c33feb8236','797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6'),
 'iclr2027_conference.bst': ('a85a0087d13bb3d19ba2ca53cae3a9db47119853','2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5'),
}
for name, (blob, sha256) in styles.items():
    out = Path('paper/visual_lens')/name
    if out.exists():
        content = out.read_bytes()
    else:
        url = f'https://api.github.com/repos/ICLR/Master-Template/git/blobs/{blob}'
        req = urllib.request.Request(url, headers={'User-Agent':'StateAdaptation-migration'})
        with urllib.request.urlopen(req, timeout=60) as response:
            obj = json.load(response)
        if obj.get('sha') != blob or obj.get('encoding') != 'base64':
            raise SystemExit(f'Unexpected upstream style response: {name}')
        content = base64.b64decode(obj['content'])
    if hashlib.sha256(content).hexdigest() != sha256:
        raise SystemExit(f'Official style mismatch: {name}')
    out.write_bytes(content)

manifest = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files}
manifest.update({f'paper/visual_lens/{n}': sha for n, (_,sha) in styles.items()})
Path('paper/visual_lens/SOURCE_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Installed {len(files)} manuscript source files and {len(styles)} verified styles.')
