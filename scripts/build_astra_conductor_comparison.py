#!/usr/bin/env python3
"""Build the four-column Astra direct vs. conductor comparison workbook."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


ROOT = Path(__file__).resolve().parents[1]
DIRECT_PATH = ROOT / "outputs/gpt-6-benchmark/results.jsonl"
CONDUCTOR_PATH = ROOT / "outputs/gpt-6-eval-frontier-clarification/results.jsonl"
OUTPUT_PATH = ROOT / "reports/astra_direct_vs_conductor_frontier_clarification.xlsx"
CSV_PATH = ROOT / "reports/astra_direct_vs_conductor_frontier_clarification.csv"


# Human summaries for every changed verdict. Keys are Astra-direct benchmark positions.
CHANGE_NOTES = {
    1: "Improvement: Astra interpreted the two holes as total and answered 4; the conductor used two holes per patch and answered the judged-correct 6.",
    16: "Regression: Astra chose B and was accepted. The conductor chose G; its judge said this conflicts with the grading standard's 2PI-vertex/propagator answer.",
    17: "Regression: Astra obtained about 457 ns (roughly 30x experiment), choice A. The conductor's angular/degeneracy treatment gave about 151 ns (roughly 10x), choice G.",
    23: "Regression: Astra's 1.38 m/s^2 answer matched the reference. The conductor treated the geometry as inconsistent and did not provide the required numerical answer.",
    28: "Improvement: Astra had the right magnitude but wrong sign (+0.08 eV). The conductor correctly returned the bound-state energy -0.08 eV.",
    37: "Both answered B. The direct judge accepted B as scientifically correct and supported by the reference explanation; the conductor judge rejected the same answer against the conflicting gold label A.",
    38: "Regression: Astra used the reference's tensor parsing and generator normalization and matched the gold formula. The conductor changed both conventions and produced a different norm.",
    48: "Both answered B. The conductor judge correctly noted that the question asks only for the exterior region, where B's expressions are correct; the direct judge graded against gold E despite the reference/sign inconsistency.",
    52: "Regression: Astra solved the Hückel determinant and gave all four accepted energies. The conductor response was truncated before the roots or final energies.",
    78: "Improvement: Astra chose Si based on a detector-artifact interpretation. The conductor chose W, matching the pure-tungsten sample and gold answer.",
    89: "Regression: Astra returned the accepted count 25. The conductor returned 18, contrary to the reference/OEIS count.",
    95: "Scientifically equivalent result, inconsistent grading: Astra's rounded -4.180 was rejected against an impossible gold value, while the conductor's -4.179556 was accepted after identifying the reference ODE error.",
    98: "Regression: Astra directly stated the requested Grassmann integral value 0. The conductor discussed the Berezin rules but declined to give the required scalar answer.",
    100: "Improvement due to reference correction: the conductor correctly said the repetition-code space is a valid stabilizer code. The direct run's same scientific conclusion was rejected against the erroneous gold 'No'.",
    113: "Improvement: both recognized non-uniqueness, but the conductor also supplied the exact gold expression under the intended real-wavefunction assumption, so its answer was accepted.",
    123: "Regression: Astra answered infinity under the boundary conditions as stated. The conductor imposed factorization-compatible domains and answered at most one, which the judge rejected.",
    143: "Both answered 1270. The conductor judge identified the gold's arithmetic error (2440 + 218 = 2658) and accepted 1270; the direct judge rejected it against gold 1260.",
    148: "Improvement due to reference correction: the conductor used perpendicular distance 1 m to each wire and obtained mu0 I/pi. Astra derived the same physical value but was rejected against the erroneous gold using sqrt(2).",
    151: "Regression: Astra returned the complete accepted set I-III-VI. The conductor returned only III and omitted two true statements.",
    152: "Improvement: Astra refused the multiple-choice premise as underdetermined. The conductor selected C, the intended Elitzur-Vaidman/polarizer test.",
    166: "Improvement: Astra chose an unilluminated tube (D). The conductor chose the judged-correct white-light tube (B), since unpolarized input does not create the proposed side pattern.",
    168: "Improvement due to reference correction: the conductor correctly placed the first correction to nu at one loop, O(u) or O(epsilon). The gold's O(epsilon^2) belongs to eta; Astra's same conclusion was rejected strictly against that gold.",
    173: "Regression caused by answer framing: Astra selected F and kept the prefactor caveat secondary. The conductor said none of the choices was exact and called F only the apparent intent, so it was rejected.",
    175: "Improvement: Astra said the sign was not unique. The conductor gave the intended decrease in N_eff and explained energy transfer from energetic daughter neutrinos to the electromagnetic plasma.",
    182: "Near-identical numerics, inconsistent tolerance: Astra's 0.295 and 1.361 were accepted as equivalent to 0.29 and 1.36; the conductor's same estimates were rejected because 0.295 would round to 0.30 at two decimals.",
    183: "Improvement due to reference correction: the conductor correctly evolved the circuit and found probability 1/4. Astra's same result was rejected against the erroneous gold 0.",
    190: "Both answered 79. Direct Astra received manual credit because the literal question asks for contracted Gaussian functions; the conductor was rejected against gold 54, which answers a different virtual-orbital question.",
    201: "Both answered E (5/3). The conductor judge correctly applied Lame's solution and flagged gold J as inconsistent with b=2a; the direct judge rejected the identical answer against that gold.",
}


def load_latest(path: Path) -> dict[str, dict]:
    """Load JSONL by example ID, retaining the last record for appended reruns."""
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["example_id"]] = record
    return records


def clean_answer(value: object, limit: int = 190) -> str:
    text = " ".join(str(value or "no answer").split())
    text = re.sub(r"^\\?\(?FINAL:\s*", "", text, flags=re.I)
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def normalized_answer(value: object) -> str:
    text = str(value or "").casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def unchanged_note(direct: dict, conductor: dict) -> str:
    verdict = "correct" if direct["correct"] else "incorrect"
    d_answer = clean_answer(direct.get("extracted_answer"))
    c_answer = clean_answer(conductor.get("extracted_answer"))
    if normalized_answer(d_answer) == normalized_answer(c_answer):
        return f"No recorded change: both were {verdict} and both answered {d_answer}."
    return (
        f"No recorded change: both were {verdict}. "
        f"Astra answered {d_answer}; conductor answered {c_answer}."
    )


def build_rows() -> list[list[object]]:
    direct = load_latest(DIRECT_PATH)
    conductor = load_latest(CONDUCTOR_PATH)
    if set(direct) != set(conductor):
        missing_direct = sorted(set(conductor) - set(direct))
        missing_conductor = sorted(set(direct) - set(conductor))
        raise RuntimeError(
            f"Run membership differs; missing direct={missing_direct}, "
            f"missing conductor={missing_conductor}"
        )

    rows: list[list[object]] = []
    for example_id, d_record in sorted(
        direct.items(), key=lambda item: item[1]["benchmark_position"]
    ):
        c_record = conductor[example_id]
        index = d_record["benchmark_position"]
        if d_record["correct"] != c_record["correct"]:
            note = CHANGE_NOTES.get(index)
            if note is None:
                raise RuntimeError(f"Missing change note for question {index}")
        else:
            note = unchanged_note(d_record, c_record)
        rows.append(
            [
                index,
                "Correct" if d_record["correct"] else "Incorrect",
                "Correct" if c_record["correct"] else "Incorrect",
                note,
            ]
        )

    if len(rows) != 202:
        raise RuntimeError(f"Expected 202 paired questions, found {len(rows)}")
    changed = sum(row[1] != row[2] for row in rows)
    if changed != 28:
        raise RuntimeError(f"Expected 28 changed verdicts, found {changed}")
    return rows


def xml_bytes(element: Element) -> bytes:
    return b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + tostring(
        element, encoding="utf-8"
    )


def inline_cell(parent: Element, ref: str, value: object, style: int = 0) -> None:
    cell = SubElement(parent, "c", {"r": ref, "t": "inlineStr", "s": str(style)})
    inline = SubElement(cell, "is")
    text = SubElement(inline, "t", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
    text.text = str(value)


def number_cell(parent: Element, ref: str, value: int, style: int = 0) -> None:
    cell = SubElement(parent, "c", {"r": ref, "s": str(style)})
    SubElement(cell, "v").text = str(value)


def worksheet_xml(rows: list[list[object]]) -> bytes:
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = Element("worksheet", {"xmlns": ns})
    views = SubElement(sheet, "sheetViews")
    view = SubElement(views, "sheetView", {"workbookViewId": "0"})
    SubElement(
        view,
        "pane",
        {"ySplit": "1", "topLeftCell": "A2", "activePane": "bottomLeft", "state": "frozen"},
    )
    SubElement(sheet, "sheetFormatPr", {"defaultRowHeight": "15"})
    cols = SubElement(sheet, "cols")
    for idx, width in enumerate((17, 25, 27, 120), start=1):
        SubElement(
            cols,
            "col",
            {"min": str(idx), "max": str(idx), "width": str(width), "customWidth": "1"},
        )

    data = SubElement(sheet, "sheetData")
    header = SubElement(data, "row", {"r": "1", "ht": "28", "customHeight": "1"})
    headers = (
        "Question Index",
        "Astra Correctness (What was recorded)",
        "Conductor Correctness",
        "Notes",
    )
    for col, value in zip("ABCD", headers):
        inline_cell(header, f"{col}1", value, 1)

    for excel_row, row in enumerate(rows, start=2):
        changed = row[1] != row[2]
        if row[1] == "Incorrect" and row[2] == "Correct":
            note_style = 5  # improvement
        elif row[1] == "Correct" and row[2] == "Incorrect":
            note_style = 6  # regression
        else:
            note_style = 4
        row_el = SubElement(
            data,
            "row",
            {"r": str(excel_row), "ht": "42" if changed else "32", "customHeight": "1"},
        )
        number_cell(row_el, f"A{excel_row}", int(row[0]), 2)
        inline_cell(row_el, f"B{excel_row}", row[1], 3 if row[1] == "Correct" else 7)
        inline_cell(row_el, f"C{excel_row}", row[2], 3 if row[2] == "Correct" else 7)
        inline_cell(row_el, f"D{excel_row}", row[3], note_style)

    SubElement(sheet, "autoFilter", {"ref": f"A1:D{len(rows) + 1}"})
    SubElement(
        sheet,
        "pageMargins",
        {"left": "0.25", "right": "0.25", "top": "0.5", "bottom": "0.5", "header": "0.2", "footer": "0.2"},
    )
    return xml_bytes(sheet)


def styles_xml() -> bytes:
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root = Element("styleSheet", {"xmlns": ns})
    fonts = SubElement(root, "fonts", {"count": "3"})
    font = SubElement(fonts, "font")
    SubElement(font, "sz", {"val": "11"})
    SubElement(font, "name", {"val": "Calibri"})
    header_font = SubElement(fonts, "font")
    SubElement(header_font, "b")
    SubElement(header_font, "color", {"rgb": "FFFFFFFF"})
    SubElement(header_font, "sz", {"val": "11"})
    SubElement(header_font, "name", {"val": "Calibri"})
    bold_font = SubElement(fonts, "font")
    SubElement(bold_font, "b")
    SubElement(bold_font, "sz", {"val": "11"})
    SubElement(bold_font, "name", {"val": "Calibri"})

    fills = SubElement(root, "fills", {"count": "7"})
    SubElement(SubElement(fills, "fill"), "patternFill", {"patternType": "none"})
    SubElement(SubElement(fills, "fill"), "patternFill", {"patternType": "gray125"})
    for color in ("FF1F4E78", "FFE2F0D9", "FFF2F2F2", "FFC6EFCE", "FFFCE4D6"):
        fill = SubElement(fills, "fill")
        pattern = SubElement(fill, "patternFill", {"patternType": "solid"})
        SubElement(pattern, "fgColor", {"rgb": color})
        SubElement(pattern, "bgColor", {"indexed": "64"})

    borders = SubElement(root, "borders", {"count": "2"})
    SubElement(borders, "border")
    border = SubElement(borders, "border")
    for side in ("left", "right", "top", "bottom"):
        side_el = SubElement(border, side, {"style": "thin"})
        SubElement(side_el, "color", {"rgb": "FFD9E1F2"})

    SubElement(root, "cellStyleXfs", {"count": "1"}).append(Element("xf", {"numFmtId": "0", "fontId": "0", "fillId": "0", "borderId": "0"}))
    cell_xfs = SubElement(root, "cellXfs", {"count": "8"})
    xfs = [
        (0, 0, 0, None),
        (1, 2, 1, {"horizontal": "center", "vertical": "center", "wrapText": "1"}),
        (0, 0, 1, {"horizontal": "center", "vertical": "top"}),
        (2, 3, 1, {"horizontal": "center", "vertical": "top"}),
        (0, 4, 1, {"vertical": "top", "wrapText": "1"}),
        (0, 5, 1, {"vertical": "top", "wrapText": "1"}),
        (0, 6, 1, {"vertical": "top", "wrapText": "1"}),
        (2, 6, 1, {"horizontal": "center", "vertical": "top"}),
    ]
    for font_id, fill_id, border_id, alignment in xfs:
        attrs = {
            "numFmtId": "0",
            "fontId": str(font_id),
            "fillId": str(fill_id),
            "borderId": str(border_id),
            "xfId": "0",
            "applyBorder": "1",
        }
        if fill_id:
            attrs["applyFill"] = "1"
        xf = SubElement(cell_xfs, "xf", attrs)
        if alignment:
            xf.set("applyAlignment", "1")
            SubElement(xf, "alignment", alignment)
    return xml_bytes(root)


def write_xlsx(rows: list[list[object]]) -> None:
    content_types = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    rels = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    workbook = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView/></bookViews>
  <sheets><sheet name="Comparison" sheetId="1" r:id="rId1"/></sheets>
  <calcPr calcId="191029"/>
</workbook>'''
    workbook_rels = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Astra direct vs conductor frontier clarification</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>'''.encode()
    app = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Excel Compatible</Application>
</Properties>'''

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml(rows))


def write_csv(rows: list[list[object]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Question Index",
                "Astra Correctness (What was recorded)",
                "Conductor Correctness",
                "Notes",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_xlsx(rows)
    write_csv(rows)
    print(OUTPUT_PATH)
    print(CSV_PATH)


if __name__ == "__main__":
    main()
