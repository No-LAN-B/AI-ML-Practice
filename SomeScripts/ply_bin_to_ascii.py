#!/usr/bin/env python3
import sys, os, re
import numpy as np

CHUNK = 20_000  # tune for your I/O; 20k rows/flush is reasonable

def find_header_end(path, sniff=1<<20):
    with open(path, "rb") as f:
        blob = f.read(sniff)
    i = blob.find(b"end_header")
    if i < 0:
        raise RuntimeError("end_header not found (sniffed %d bytes)" % sniff)
    j = i + len(b"end_header")
    # include trailing newline(s) if present (handles \n or \r\n)
    if j < len(blob) and blob[j:j+1] in (b"\n", b"\r"):
        j += 1
        if j < len(blob) and blob[j:j+1] == b"\n":
            j += 1
    return j, blob[:j].decode("ascii", errors="ignore")

def parse_counts_and_props(header_text):
    # Extract vertex count and number of float properties for vertex element
    lines = [ln.strip() for ln in header_text.splitlines()]
    in_vertex = False
    vertex_count = None
    prop_count = 0
    for ln in lines:
        if ln.startswith("element "):
            in_vertex = ln.startswith("element vertex")
            if in_vertex:
                parts = ln.split()
                vertex_count = int(parts[2])
        elif in_vertex and ln.startswith("property "):
            # Expect: property float <name>
            parts = ln.split()
            if len(parts) >= 3 and parts[1].startswith("float"):
                prop_count += 1
            else:
                raise RuntimeError("This script expects all vertex props to be float; saw: %r" % ln)
        elif ln.startswith("end_header"):
            break
    if vertex_count is None or prop_count == 0:
        raise RuntimeError("Could not parse vertex element/props from header.")
    return vertex_count, prop_count

def make_ascii_header(header_text):
    out_lines = []
    for ln in header_text.splitlines():
        if ln.startswith("format "):
            out_lines.append("format ascii 1.0")
        else:
            out_lines.append(ln)
        if ln.strip() == "end_header":
            break
    # Ensure a single trailing newline after end_header
    if not out_lines[-1].endswith("\n"):
        out_lines[-1] = out_lines[-1] + "\n"
    return "\n".join(out_lines)

def main(inp, outp):
    hb, header_text = find_header_end(inp)
    N, P = parse_counts_and_props(header_text)   # vertices, float props per vertex
    stride_bytes = P * 4
    ascii_header = make_ascii_header(header_text)

    filesize = os.path.getsize(inp)
    expected = hb + N * stride_bytes
    if filesize < expected:
        raise RuntimeError(f"File too small: {filesize} < expected {expected} (hb={hb}, N={N}, P={P})")
    if filesize > expected:
        # Some exporters append extras; we’ll ignore trailing bytes but warn.
        print(f"[warn] File larger than expected: {filesize} > {expected}. Proceeding.", file=sys.stderr)

    # Zero-copy float32 view of the whole vertex block: shape (N, P)
    mm = np.memmap(inp, mode="r", dtype="<f4", offset=hb, shape=(N, P))

    with open(outp, "w", newline="\n") as w:
        w.write(ascii_header)  # same schema, ascii format
        # Stream rows
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            block = mm[start:end]  # (chunk, P) float32
            # Convert to strings efficiently
            # Map each row to space-separated string; use repr-like with moderate precision
            for row in block:
                # 6–7 sig figs is typical for PLY ASCII; adjust if you need more.
                s = " ".join(f"{v:.7g}" for v in row.tolist())
                w.write(s + "\n")
            # Optional progress
            if (start // CHUNK) % 50 == 0:
                print(f"[info] wrote rows {start}..{end-1}", file=sys.stderr)

    print(f"[done] Wrote ASCII PLY: {outp}  (vertices={N}, props/vertex={P})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ply_bin_to_ascii.py input.ply output_ascii.ply", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
