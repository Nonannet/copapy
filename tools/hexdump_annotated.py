#!/usr/bin/env python3
"""Annotated hex dump of a COFF/PE object file.

Dumps the *entire* file (COFF header, section headers, section contents,
relocations, line numbers, symbol table and string table) as a hex+ASCII
listing.  Every byte range is annotated with the region it belongs to, and
symbols located inside a section are marked at their file offset.  When
objdump is available its disassembly (`objdump -d`) is woven into the body:
code-section bytes are shown one instruction per line (offset, hex, ASCII and
mnemonic), while any non-instruction bytes and data regions stay as hex rows.

Usage:
    python tools/hexdump_annotated.py <input.obj> <output.hex>
    python tools/hexdump_annotated.py -r|--reassemble <input.hex> <output.obj>

The disassembly is produced with `wsl objdump -d` (Windows + WSL only); if
objdump is unavailable the hex dump is still written, just without disassembly.
With --reassemble the tool runs in reverse: it reads a hex dump produced here
and rebuilds the original object file from the byte values (the ASCII column
and mnemonics are ignored; only the hex bytes and their offsets are used).
"""

import bisect
import re
import struct
import subprocess
import sys
from typing import Any, Optional

IMAGE_FILE_HEADER_SIZE = 20
IMAGE_SECTION_HEADER_SIZE = 40
IMAGE_SYMBOL_SIZE = 18
IMAGE_RELOCATION_SIZE = 10
IMAGE_LINENUMBER_SIZE = 6

SKIP_STORAGE = {0, 103, 104}  # NULL, FILE, SECTION

P_META = 10
P_SYM = 20
P_RELOC = 30
P_SECTION = 40


def cstr(buf: bytes, off: int) -> str:
    out = bytearray()
    while off < len(buf) and buf[off] != 0:
        out.append(buf[off])
        off += 1
    return out.decode("latin-1")


def short_name(raw: bytes) -> str:
    idx = raw.find(b"\x00")
    if idx >= 0:
        raw = raw[:idx]
    return raw.decode("latin-1")


def section_display_name(raw: bytes, string_table: bytes) -> str:
    text = short_name(raw)
    if text.startswith("/"):
        try:
            off = int(text[1:])
        except ValueError:
            return text
        return cstr(string_table, off) if string_table else text
    return text


def symbol_name(raw: bytes, string_table: bytes) -> str:
    if raw[:4] == b"\x00\x00\x00\x00":
        off = struct.unpack_from("<I", raw, 4)[0]
        return cstr(string_table, off) if string_table else "@str%d" % off
    return short_name(raw[:8])


def fh_from(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < IMAGE_FILE_HEADER_SIZE:
        return None
    (machine, nsec, ts, sym_ptr, sym_count, opt, _chars) = \
        struct.unpack_from("<HHIIIHH", buf, 0)
    return {
        "Machine": machine,
        "NumberOfSections": nsec,
        "TimeDateStamp": ts,
        "PointerToSymbolTable": sym_ptr,
        "NumberOfSymbols": sym_count,
        "SizeOfOptionalHeader": opt,
    }


def read_string_table(buf: bytes, sym_ptr: int, sym_count: int,
                      size: int) -> tuple[bytes, int]:
    start = sym_ptr + sym_count * IMAGE_SYMBOL_SIZE
    if sym_ptr == 0 or sym_count == 0 or start >= size:
        return b"", 0
    str_size = struct.unpack_from("<I", buf, start)[0]
    if str_size < 4:
        return b"", start
    return buf[start:start + min(str_size, size - start)], start


def parse_sections(buf: bytes, fh: dict[str, int], string_table: bytes,
                   size: int) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    base = IMAGE_FILE_HEADER_SIZE + fh["SizeOfOptionalHeader"]
    for i in range(fh["NumberOfSections"]):
        o = base + i * IMAGE_SECTION_HEADER_SIZE
        if o + IMAGE_SECTION_HEADER_SIZE > size:
            break
        (vsize, vaddr, raw_size, raw_ptr, reloc_ptr, lineno_ptr,
         nreloc, nlineno, _ch) = struct.unpack_from("<IIIIIIHHI", buf, o + 8)
        sections.append({
            "index": i + 1,
            "name": section_display_name(buf[o:o + 8], string_table),
            "SizeOfRawData": raw_size,
            "PointerToRawData": raw_ptr,
            "PointerToRelocations": reloc_ptr,
            "NumberOfRelocations": nreloc,
            "PointerToLinenumbers": lineno_ptr,
            "NumberOfLinenumbers": nlineno,
        })
    return sections


def parse_symbols(buf: bytes, string_table: bytes, sym_ptr: int, sym_count: int,
                  size: int) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    if not sym_ptr or sym_count == 0:
        return result
    i, off = 0, sym_ptr
    while i < sym_count and off + IMAGE_SYMBOL_SIZE <= size:
        raw = buf[off:off + IMAGE_SYMBOL_SIZE]
        name = symbol_name(raw, string_table)
        value, secnum, _type, sclass, naux = struct.unpack_from("<IhHBB", raw, 8)
        if name and sclass not in SKIP_STORAGE and secnum > 0:
            result.setdefault(name, (secnum, value))
        i += 1 + naux
        off += IMAGE_SYMBOL_SIZE * (1 + naux)
    return result


def build_regions(size: int, fh: Optional[dict[str, int]],
                  sections: list[dict[str, Any]], str_start: int,
                  str_len: int) -> list[tuple[int, int, int, str]]:
    regions: list[tuple[int, int, int, str]] = \
        [(0, IMAGE_FILE_HEADER_SIZE, P_META, "COFF file header")]

    if fh is not None:
        base = IMAGE_FILE_HEADER_SIZE + fh["SizeOfOptionalHeader"]
        if fh["NumberOfSections"]:
            regions.append((base,
                            base + fh["NumberOfSections"] * IMAGE_SECTION_HEADER_SIZE,
                            P_META, "section headers"))

        for sec in sections:
            name = sec["name"]
            if sec["PointerToRawData"] and sec["SizeOfRawData"]:
                regions.append((sec["PointerToRawData"],
                                sec["PointerToRawData"] + sec["SizeOfRawData"],
                                P_SECTION, "section %s" % name))
            if sec["PointerToRelocations"] and sec["NumberOfRelocations"]:
                regions.append((sec["PointerToRelocations"],
                                sec["PointerToRelocations"] +
                                sec["NumberOfRelocations"] * IMAGE_RELOCATION_SIZE,
                                P_RELOC, "relocations for %s" % name))
            if sec["PointerToLinenumbers"] and sec["NumberOfLinenumbers"]:
                regions.append((sec["PointerToLinenumbers"],
                                sec["PointerToLinenumbers"] +
                                sec["NumberOfLinenumbers"] * IMAGE_LINENUMBER_SIZE,
                                P_RELOC, "line numbers for %s" % name))

    if fh is not None and fh["PointerToSymbolTable"] and fh["NumberOfSymbols"]:
        regions.append((fh["PointerToSymbolTable"],
                        fh["PointerToSymbolTable"] +
                        fh["NumberOfSymbols"] * IMAGE_SYMBOL_SIZE,
                        P_SYM, "symbol table"))
    if str_start and str_len:
        regions.append((str_start, str_start + str_len, P_SYM, "string table"))

    cleaned: list[tuple[int, int, int, str]] = []
    for start, end, prio, label in regions:
        if start < 0 or end <= start or start >= size:
            continue
        cleaned.append((start, min(end, size), prio, label))
    return cleaned


def assign_owners(size: int, regions: list[tuple[int, int, int, str]]
                  ) -> list[Optional[str]]:
    owners: list[Optional[str]] = [None] * size
    prio = [0] * size
    for start, end, p, label in regions:
        for i in range(start, end):
            if p >= prio[i]:
                prio[i] = p
                owners[i] = label
    return owners


def segments_from_owners(owners: list[Optional[str]]
                         ) -> list[tuple[int, int, Optional[str]]]:
    segs: list[tuple[int, int, Optional[str]]] = []
    start = 0
    for i in range(1, len(owners) + 1):
        if i == len(owners) or owners[i] != owners[start]:
            segs.append((start, i, owners[start]))
            start = i
    return segs


def hex_group(chunk: bytes) -> str:
    lh = " ".join("%02X" % b for b in chunk[:8]).ljust(8 * 3 - 1)
    rh = " ".join("%02X" % b for b in chunk[8:16]).ljust(8 * 3 - 1)
    return "%s  %s" % (lh, rh)


SEC_HEADER_RE = re.compile(r"^Disassembly of section (\S+):\s*$")
SYM_LINE_RE = re.compile(r"^[0-9a-f]+ <(.+)>:\s*$")
INSN_RE = re.compile(r"^\s*([0-9a-f]+):\t(.*?)\t(.*?)\s*$")

Disasm = dict[str, list[tuple[int, list[int], str]]]

HEX_BODY_RE = re.compile(r"^\s*([0-9a-fA-F]{8})\s{2}(.*?)\s{2}\|")
HEX_TOKEN_RE = re.compile(r"[0-9a-fA-F]{2}")
DUMP_SIZE_RE = re.compile(r"size=(\d+)\s+bytes")


def run_objdump(obj_path: str) -> Optional[str]:
    """Return `wsl objdump -d` output, or None if it is unavailable/fails."""
    try:
        r = subprocess.run(["wsl", "objdump", "-d", obj_path.replace("\\", "/")],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def parse_objdump_disasm(text: str) -> Disasm:
    """Parse `objdump -d` output -> {section_name: [(addr, bytes, mnemonic), ..]}."""
    sections: Disasm = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = SEC_HEADER_RE.match(line)
        if m:
            current = m.group(1)
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        if SYM_LINE_RE.match(line):
            continue
        m = INSN_RE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        byte_tokens = m.group(2).split()
        mnemonic = m.group(3).strip()
        sections[current].append((addr, [int(b, 16) for b in byte_tokens], mnemonic))
    return sections


def reassemble(in_path: str, out_path: str) -> int:
    """Rebuild an object file from a hex dump produced by this tool.

    Only the hex byte values and their offsets are used; the ASCII column,
    mnemonics and annotation comments are ignored.
    """
    with open(in_path, "r", encoding="latin-1") as f:
        lines = f.read().splitlines()

    cells: dict[int, int] = {}
    conflicts: list[int] = []
    expected: Optional[int] = None

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(";"):
            if expected is None:
                m = DUMP_SIZE_RE.search(stripped)
                if m:
                    expected = int(m.group(1))
            continue
        m = HEX_BODY_RE.match(line)
        if not m:
            continue
        off = int(m.group(1), 16)
        for k, tok in enumerate(HEX_TOKEN_RE.findall(m.group(2))):
            pos, val = off + k, int(tok, 16)
            if pos in cells and cells[pos] != val:
                conflicts.append(pos)
            cells[pos] = val

    if not cells:
        sys.stderr.write("no hex data lines found in %s\n" % in_path)
        return 1

    size = max(cells) + 1
    missing = [i for i in range(size) if i not in cells]
    if missing:
        sys.stderr.write("hex dump is incomplete: %d byte(s) missing (first at 0x%X)\n"
                         % (len(missing), missing[0]))
        return 1
    if conflicts:
        sys.stderr.write("conflicting byte values at %d offset(s) (first at 0x%X)\n"
                         % (len(conflicts), conflicts[0]))
        return 1
    if expected is not None and expected != size:
        sys.stderr.write("size mismatch: dump header says %d bytes, data covers %d\n"
                         % (expected, size))
        return 1

    with open(out_path, "wb") as f:
        f.write(bytes(cells[i] for i in range(size)))

    print("Reassembled %d bytes to %s from %s"
          % (size, out_path.replace("\\", "/"), in_path.replace("\\", "/")))
    return 0


def main(argv: list[str]) -> int:
    args = argv[1:]

    if any(a in ("-r", "--reassemble") for a in args):
        rest = [a for a in args if a not in ("-r", "--reassemble")]
        if len(rest) != 2:
            sys.stderr.write(__doc__ or "")
            return 2
        return reassemble(rest[0], rest[1])

    if len(args) < 2:
        sys.stderr.write(__doc__ or "")
        return 2

    in_path, out_path = args[0], args[1]

    with open(in_path, "rb") as f:
        buf = f.read()
    size = len(buf)

    fh = fh_from(buf)
    sym_ptr = fh["PointerToSymbolTable"] if fh else 0
    sym_count = fh["NumberOfSymbols"] if fh else 0
    string_table, str_start = read_string_table(buf, sym_ptr, sym_count, size)
    sections = parse_sections(buf, fh, string_table, size) if fh else []

    sym_map = parse_symbols(buf, string_table, sym_ptr, sym_count, size)
    by_index = {s["index"]: s for s in sections}
    sym_offsets: dict[int, list[str]] = {}
    for name, (secnum, value) in sym_map.items():
        sec = by_index.get(secnum)
        if sec and sec["PointerToRawData"]:
            fo = sec["PointerToRawData"] + value
            if 0 <= fo < size:
                sym_offsets.setdefault(fo, [])
                if name not in sym_offsets[fo]:
                    sym_offsets[fo].append(name)

    regions = build_regions(size, fh, sections, str_start, len(string_table))
    owners = assign_owners(size, regions)
    base_by_name = {s["name"]: s["PointerToRawData"] for s in sections
                    if s["PointerToRawData"]}

    text = run_objdump(in_path)
    disasm = parse_objdump_disasm(text) if text is not None else {}

    insn_at: dict[int, tuple[list[int], str]] = {}
    for sec_name, insns in disasm.items():
        base = base_by_name.get(sec_name)
        if base is None:
            continue
        for addr, ibytes, mnemonic in insns:
            insn_at[base + addr] = (ibytes, mnemonic)
    insn_starts = sorted(insn_at)
    insn_w = max((len(b) for b, _ in insn_at.values()), default=0)
    insn_w = max(insn_w, 1) * 3 - 1

    out = ["; annotated hex dump of %s" % in_path.replace("\\", "/")]
    if fh:
        arch = "x86_64" if fh["Machine"] == 0x8664 else "0x%04X" % fh["Machine"]
        out.append("; machine=%s  sections=%d  symbols=%d  time=0x%08X  size=%d bytes"
                   % (arch, fh["NumberOfSections"], fh["NumberOfSymbols"],
                      fh["TimeDateStamp"], size))
        out.append("; sections: " + (", ".join(
            "%s[@0x%X sz=0x%X]" % (s["name"], s["PointerToRawData"], s["SizeOfRawData"])
            for s in sections) or "(none)"))
    else:
        out.append("; not a COFF object (header too small); size=%d bytes" % size)
    if text is not None:
        out.append("; objdump: wsl objdump -d %s" % in_path.replace("\\", "/"))
    else:
        out.append("; objdump: wsl objdump not available (hex only, no disassembly)")
    out.append("")

    for seg_start, seg_end, label in segments_from_owners(owners):
        out.append("; ---- %s  @0x%08X..0x%08X (%d bytes) ----"
                   % (label or "unmapped data", seg_start, seg_end, seg_end - seg_start))
        i = seg_start
        while i < seg_end:
            entry = insn_at.get(i)
            if entry is not None:
                ibytes, mnemonic = entry
                hexs = " ".join("%02X" % b for b in ibytes).ljust(insn_w)
                ascii_ = "".join(chr(b) if 0x20 <= b <= 0x7E else "." for b in ibytes)
                line = "%08X  %s  |%s|  %s" % (i, hexs, ascii_, mnemonic)
                if i in sym_offsets:
                    line += "   ; " + ", ".join(sym_offsets[i])
                out.append(line)
                i += len(ibytes) if ibytes else 1
                continue
            nxt = seg_end
            j = bisect.bisect_right(insn_starts, i)
            if j < len(insn_starts) and insn_starts[j] < nxt:
                nxt = insn_starts[j]
            chunk = buf[i:min(i + 16, nxt)]
            ascii_ = "".join(chr(b) if 0x20 <= b <= 0x7E else "." for b in chunk)
            line = "%08X  %s  |%s|" % (i, hex_group(chunk), ascii_)
            notes = [n for t in range(i, i + len(chunk)) for n in sym_offsets.get(t, [])]
            if notes:
                line += "   ; " + ", ".join(notes)
            out.append(line)
            i += len(chunk)

    with open(out_path, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")

    print("Annotated hex dump written to %s (%d regions, %d bytes)"
          % (out_path.replace("\\", "/"), len(regions), size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
