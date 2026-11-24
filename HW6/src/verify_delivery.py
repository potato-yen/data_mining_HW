"""Cross-check exported files and PDFKit text after rendering both PDFs."""
import csv
import hashlib
import json
import math
import re
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT/'results/adaboost.json').read_text())
    assert hashlib.sha256((ROOT/'AdaBoost_學習單.docx').read_bytes()).hexdigest() == result['source_sha256']
    expected = [[2, 8, 5, 5, 3, 7], [5, 5, 6, 4, 1, 9], [3, 7, 9, 1, 6, 4]]
    with (ROOT/'results/candidate_errors.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    assert [int(row['sample_errors']) for row in rows] == sum(expected, [])
    with (ROOT/'results/weights.csv').open() as stream:
        weights = list(csv.DictReader(stream))
    assert len(weights) == 20
    for t in (1, 2):
        actual = [float(row['weight_after']) for row in weights if int(row['round']) == t]
        assert math.isclose(sum(actual), 1, abs_tol=1e-12)
        assert actual == result['rounds'][t-1]['weights_after']
    with (ROOT/'results/predictions.csv').open() as stream:
        predictions = list(csv.DictReader(stream))
    assert [int(p['prediction']) for p in predictions] == [-1, 1, -1, 1]
    pdfs = {}
    for stem, expected_pages in [('worksheet', 4), ('main', 5)]:
        inspection = json.loads((ROOT/f'tmp/qa/{stem}/inspection.json').read_text())
        assert inspection['page_count'] == expected_pages
        text = '\n'.join(inspection['page_texts'])
        assert all(len(page.strip()) > 100 for page in inspection['page_texts'])
        assert '??' not in text
        for r in result['rounds']:
            assert f'{r["epsilon"]:.6f}' in text
            assert f'{r["alpha"]:.6f}' in text
            if r['weights_after']:
                for w in r['weights_after']:
                    assert f'{w:.6f}' in text
        for p in result['predictions']:
            assert f'{abs(p["score"]):.6f}' in text
        assert all(q in text for q in ('Q1.', 'Q2.', 'Q3.'))
        log = (ROOT/f'output/pdf/{stem}.log').read_text()
        for problem in ('Overfull', 'Missing character:', 'undefined references'):
            assert problem not in log, (stem, problem)
        pdf = ROOT/f'output/pdf/{stem}.pdf'
        pdf_bytes = pdf.read_bytes()
        # xdvipdfmx compresses font descriptor objects inside Flate streams.
        objects = [pdf_bytes]
        for match in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', pdf_bytes, re.S):
            try:
                objects.append(zlib.decompress(match[1]))
            except zlib.error:
                pass  # Not every PDF stream uses Flate compression.
        assert any(b'/FontFile' in obj for obj in objects), 'No embedded font stream found'
        pdfs[stem] = dict(pages=expected_pages, bytes=pdf.stat().st_size,
                          sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                          numeric_text_check=True, embedded_font_stream=True)
    summary = dict(candidate_rows=len(rows), weight_rows=len(weights), prediction_rows=len(predictions),
                   source_sha256=result['source_sha256'], pdfs=pdfs,
                   visual_inspection='Recorded separately in docs/verification.md')
    (ROOT/'results/verification_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
