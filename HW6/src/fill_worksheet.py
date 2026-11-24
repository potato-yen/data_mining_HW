"""Fill a copy of the retained Word worksheet; preserve other package parts."""
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def set_text(paragraph, text):
    """Keep the paragraph/run formatting, replacing only its visible text."""
    nodes = list(re.finditer(r'<w:t\b[^>]*>.*?</w:t>', paragraph, re.S))
    if nodes:
        first = True
        def replace(match):
            nonlocal first
            value = text if first else ''
            first = False
            return '<w:t xml:space="preserve">'+escape(value)+'</w:t>'
        return re.sub(r'<w:t\b[^>]*>.*?</w:t>', replace, paragraph, flags=re.S)
    properties = re.search(r'<w:rPr\b[^>]*>.*?</w:rPr>', paragraph, re.S)
    run = '<w:r>'+(properties[0] if properties else '')
    run += '<w:t xml:space="preserve">'+escape(text)+'</w:t></w:r>'
    return paragraph.replace('</w:p>', run+'</w:p>')


def fill_slots(xml, values):
    assert len(re.findall(r'_{3,}', xml)) == len(values)
    values = iter(values)
    return re.sub(r'_{3,}', lambda _: escape(str(next(values))), xml)


def clone_clean(xml):
    # Cloned rows/paragraphs must not reuse Word's editing identifiers.
    return re.sub(r' w14:(?:paraId|textId)="[^"]*"', '', xml)


def weight_table(source, round_result):
    rows = list(re.finditer(r'<w:tr\b[^>]*>.*?</w:tr>', source, re.S))
    template = rows[1][0]
    def row(values):
        cells = iter(values)
        def cell(match):
            return re.sub(r'<w:p\b[^>]*>.*?</w:p>',
                          lambda p: set_text(p[0], str(next(cells))), match[0], count=1, flags=re.S)
        return clone_clean(re.sub(r'<w:tc\b[^>]*>.*?</w:tc>', cell, template, flags=re.S))
    t = round_result['round']
    values = [['ID', 'y', f'h{t}(x)', '原權重', 'u', '新權重']]
    values += [[i+1, f'{y:+d}', f'{round_result["original_predictions"][i]:+d}',
                f'{round_result["weights_before"][i]:.6f}',
                f'{round_result["unnormalized_weights"][i]:.6f}',
                f'{round_result["weights_after"][i]:.6f}']
               for i, y in enumerate([-1, -1, -1, -1, 1, 1, 1, -1, -1, 1])]
    return source[:rows[0].start()]+''.join(row(v) for v in values)+source[rows[-1].end():]


def main():
    source = ROOT/'AdaBoost_學習單.docx'
    output = ROOT/'report/AdaBoost_學習單_完成版.docx'
    result = json.loads((ROOT/'results/adaboost.json').read_text())
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == result['source_sha256']
    with ZipFile(source) as archive:
        xml = archive.read('word/document.xml').decode('utf-8')
        body = re.search(r'(<w:body>)(.*?)(</w:body>)', xml, re.S)
        blocks = list(re.finditer(r'<w:(p|tbl|sectPr)\b[^>]*>.*?</w:\1>', body[2], re.S))
        assert len(blocks) == 72, 'Unexpected source worksheet structure'
        parts = [m[0] for m in blocks]
        original = parts.copy()
        for t, (table_index, best_index, alpha_index) in enumerate([(16, 18, 19), (28, 30, 32), (41, 43, 45)]):
            r = result['rounds'][t]
            parts[table_index] = fill_slots(parts[table_index], r['sample_errors'].values())
            s = r['stump']
            parts[best_index] = fill_slots(parts[best_index], [r['rule'], s['split'], f'{s["left"]:+d}', f'{s["right"]:+d}'])
            parts[alpha_index] = fill_slots(parts[alpha_index], [f'{r["epsilon"]:.6f}', f'{r["alpha"]:.6f}'])
        parts[21] = fill_slots(parts[21], ['8、9', '1/16 = 0.062500', '1/4 = 0.250000'])
        parts[22] = set_text(parts[22],
            'ε₁ = 0.1 + 0.1 = 0.2；α₁ = ½ ln(4) ≈ 0.693147。以 ID 8 為例，'
            'y₈h₁(x₈) = −1，u₈ = 0.1 × exp(α₁) = 0.2；Σuᵢ = 0.8，'
            '故新權重為 0.2/0.8 = 0.25。下表列出原始 10 筆資料的權重更新（顯示六位小數）：')
        parts[22] += weight_table(original[4], result['rounds'][0])
        parts[22] += set_text(clone_clean(original[34]), '更新後權重總和：8 × 1/16 + 2 × 1/4 = 1。')
        parts[33] = fill_slots(parts[33], ['1'])
        parts[34] = set_text(parts[34],
            '原始資料分錯 ID 5、6、7，所以 ε₂ = 3 × 1/16 = 3/16；'
            'α₂ = ½ ln(13/3) ≈ 0.733169。抽樣集雖只錯 1 筆，不能用 1/10 取代 ε₂。')
        parts[35] = set_text(parts[35],
            '以 ID 5 為例，y₅h₂(x₅) = −1，u₅ = 0.0625 × exp(α₂) ≈ 0.130104；'
            'Σuᵢ ≈ 0.780625，故新權重 u₅/Σuᵢ = 1/6 ≈ 0.166667（中間計算未四捨五入）。')
        parts[36] = set_text(parts[36], '原始 10 筆資料在 Round 2 更新後的權重如下：')
        parts[36] += weight_table(original[4], result['rounds'][1])
        parts[36] += set_text(clone_clean(original[34]),
            '新權重：ID 1–4、10 各為 1/26，ID 5–7 各為 1/6，ID 8、9 各為 2/13；'
            '總和 = 5 × 1/26 + 3 × 1/6 + 2 × 2/13 = 1。')
        parts[46] = fill_slots(parts[46], [5])
        parts[47] = set_text(parts[47],
            '原始資料分錯 ID 1、2、3、4、10，所以 ε₃ = 5 × 1/26 = 5/26 ≈ 0.192308；'
            'α₃ = ½ ln(21/5) ≈ 0.717542。本輪抽樣錯誤只有 1 筆；依題意不再更新權重。')
        values = []
        for p in result['predictions']:
            values += [*[f'{v:+d}' for v in p['votes']], f'{p["score"]:.6f}', f'{p["prediction"]:+d}']
        parts[50] = fill_slots(parts[50], values)
        parts[51] = set_text(parts[51],
            '投票示範：x = 5.5 時，h₁ = +1、h₂ = −1、h₃ = +1。'
            'S(5.5) = ½ ln(4) − ½ ln(13/3) + ½ ln(21/5)'
            ' ≈ 0.693147 − 0.733169 + 0.717542 = 0.677521 > 0，所以 H(5.5) = +1。')
        answers = {
            54: 'ID 8、9 的權重由 0.1 升至 0.25，是分對樣本新權重 0.0625 的四倍，因此依權重有放回抽樣時較容易重複抽到。',
            55: '本例第二輪 ID 8 出現兩次、ID 9 出現三次，但較高權重不保證每次抽樣都會重複。',
            57: '抽樣資料用來選規則，ε₂ 則定義為原始 10 筆資料中錯分樣本的當輪權重總和。',
            58: '本例 E 在抽樣集只錯 1/10，但在原始資料錯分 ID 5、6、7，所以 ε₂ = 3/16。',
            60: '抽樣組成改變會改變各規則的錯誤筆數：第一輪 A 錯 2 筆、E 錯 3 筆，因此選 A。',
            61: '第二輪重複抽到 ID 8、9，使 A 錯 5 筆，而 E 只錯 1 筆，因此改選切分點 9.5 的 E。',
        }
        for i, answer in answers.items():
            parts[i] = set_text(parts[i], answer)
        # Keep the original package XML verbatim outside the edited body blocks.
        for block, replacement in reversed(list(zip(blocks, parts))):
            xml = xml[:body.start(2)+block.start()]+replacement+xml[body.start(2)+block.end():]
        root = ET.fromstring(xml)
        assert not re.search(r'_{3,}', ''.join(root.itertext())), 'Unfilled answer slot'
        output.parent.mkdir(exist_ok=True)
        with ZipFile(output, 'w') as target:
            for member in archive.infolist():
                target.writestr(member, xml.encode('utf-8') if member.filename == 'word/document.xml' else archive.read(member))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with ZipFile(source) as a, ZipFile(output) as b:
        assert a.namelist() == b.namelist()
        assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
    print(f'Completed: {output}\nVerified: all slots filled; only word/document.xml changed; source unchanged.')


if __name__ == '__main__':
    main()
