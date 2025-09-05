import sys, os, tarfile, hashlib, io, zstandard as zstd

if len(sys.argv) < 2:
    print("usage: hash_payload.py <file...>")
    sys.exit(2)

paths = sorted(sys.argv[1:])
hs = hashlib.sha256()
cctx = zstd.ZstdCompressor(level=19)
with io.BytesIO() as raw:
    with tarfile.open(fileobj=raw, mode="w|") as tar:
        for p in paths:
            ti = tarfile.TarInfo(name=p)
            st = os.stat(p)
            ti.size = st.st_size
            ti.mtime = 0
            with open(p, "rb") as f:
                tar.addfile(ti, f)
    raw.seek(0)
    data = cctx.compress(raw.getvalue())
    hs.update(data)
print("sha256:" + hs.hexdigest())

