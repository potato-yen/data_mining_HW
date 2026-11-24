"""Generate shared LaTeX answers directly from the computed results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(caption, label, headers, rows, columns=None):
    columns = columns or 'r'*len(headers)
    return '\n'.join([
        r'\begin{center}', r'\captionof{table}{'+caption+'}',
        r'\label{'+label+'}', r'\begin{tabular}{'+columns+'}',
        r'\toprule', ' & '.join(headers)+r' \\', r'\midrule',
        *[' & '.join(map(str, row))+r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{center}',
    ])+'\n'


def build_report(result):
    sections = ROOT / 'report' / 'sections'
    sections.mkdir(parents=True, exist_ok=True)
    for r in result['rounds']:
        t = r['round']
        s = r['stump']
        text = rf'\section*{{{["三", "四", "五"][t-1]}、Round {t}}}'+'\n'
        if t == 1:
            text += '初始權重皆為 $0.1$。本輪抽樣 ID 為 '
        else:
            text += f'使用 Round {t-1} 更新後的權重。固定抽樣 ID 為 '
        text += r'\['+r'['+r',\,'.join(map(str, r['sample_ids']))+r'].\]'+'\n'
        text += table(f'Round {t} 的候選規則比較', f'tab:round{t}',
                      ['規則', '切分點', '左側預測', '右側預測', '抽樣錯誤筆數'],
                      [[name, rule['split'], f"${rule['left']:+d}$", f"${rule['right']:+d}$",
                        r['sample_errors'][name]] for name, rule in result['rules'].items()])
        text += (f"最少錯誤為 {min(r['sample_errors'].values())} 筆，選出規則 {r['rule']}，"
                 '其分類器為\n'+rf'\[h_{t}(x)=\begin{{cases}}{s["left"]:+d},&x\le {s["split"]},\\'
                 +rf'{s["right"]:+d},&x>{s["split"]}.\end{{cases}}\]'+'\n')
        exact = [r'2(0.1)=\frac15', r'3\left(\frac1{16}\right)=\frac3{16}',
                 r'5\left(\frac1{26}\right)=\frac5{26}'][t-1]
        ratio = ['4', r'\frac{13}{3}', r'\frac{21}{5}'][t-1]
        text += ('回到原始 10 筆資料，分錯 ID 為 '+', '.join(map(str, r['wrong_ids']))+'，故\n'
                 +rf'\[\epsilon_{t}={exact}\approx {r["epsilon"]:.6f},\qquad '
                 +rf'\alpha_{t}=\frac12\ln\left({ratio}\right)\approx {r["alpha"]:.6f}.\]'+'\n')
        if t < 3:
            wrong_i = r['wrong_ids'][0]-1
            text += (rf'正規化常數 $Z_{t}=\sum_i u_i={r["normalizer"]:.6f}$。'
                     +f'以分錯的 ID {wrong_i+1} 為例，$y_i h_{t}(x_i)=-1$，因此\n'
                     +rf'\[u_{{{wrong_i+1}}}={r["weights_before"][wrong_i]:.6f}'
                     +rf'\exp(+{r["alpha"]:.6f})\approx {r["unnormalized_weights"][wrong_i]:.6f},\qquad '
                     +rf'w_{{{wrong_i+1}}}^{{({t+1})}}=\frac{{u_{{{wrong_i+1}}}}}{{Z_{t}}}'
                     +rf'\approx {r["weights_after"][wrong_i]:.6f}.\]'+'\n')
            if t == 1:
                text += ('分對樣本的新權重均為 $1/16=0.062500$；分錯樣本（ID 8、9）'
                         '的新權重均為 $1/4=0.250000$。\n')
            text += table(f'Round {t} 在原始資料上的權重更新', f'tab:weights{t}',
                          ['ID', '$y_i$', f'$h_{t}(x_i)$', f'$w_i^{{({t})}}$', '$u_i$', f'$w_i^{{({t+1})}}$'],
                          [[i+1, f'${row["y"]:+d}$', f'${r["original_predictions"][i]:+d}$',
                            f'{r["weights_before"][i]:.6f}', f'{r["unnormalized_weights"][i]:.6f}',
                            f'{r["weights_after"][i]:.6f}'] for i, row in enumerate(result['data'])])
            if t == 1:
                text += r'權重總和為 $8(1/16)+2(1/4)=1$；兩筆錯分樣本合計占一半抽樣機率。'+'\n'
            else:
                text += (r'更新後 ID 1--4、10 各為 $1/26$，ID 5--7 各為 $1/6$，'
                         r'ID 8、9 各為 $2/13$。因此總和為'+'\n'
                         r'\[5\left(\frac1{26}\right)+3\left(\frac16\right)+2\left(\frac2{13}\right)=1.\]'+'\n'
                         '注意：抽樣集只有 ID 5 的一筆錯誤（$1/10$）；'
                         r'原始資料上 ID 5、6、7 的權重合計才是 $\epsilon_2=3/16$。'+'\n')
        else:
            text += (r'規則 D 對原始資料分錯 ID 1、2、3、4、10，共 \textbf{5 筆}；'
                     '抽樣錯誤只有 1 筆，兩者的評估資料不同。本題不再更新第三輪權重。\n')
        (sections / f'round{t}.tex').write_text(text)

    text = r'\section*{六、三個分類器的加權投票}'+'\n'
    text += r'令 $S(x)=\sum_{t=1}^3\alpha_t h_t(x)$、$H(x)=\operatorname{sign}(S(x))$。'+'\n'
    text += table('新資料的完整預測結果', 'tab:predictions',
                  ['$x$', '$h_1(x)$', '$h_2(x)$', '$h_3(x)$', '$S(x)$', '$H(x)$'],
                  [[p['x'], *[f'${v:+d}$' for v in p['votes']], f'{p["score"]:.6f}',
                    f'${p["prediction"]:+d}$'] for p in result['predictions']])
    a = [r['alpha'] for r in result['rounds']]
    text += ('例如 $x=5.5$，三個分類器的預測依序為 $+1,-1,+1$，因此\n'
             +r'\begin{align*}S(5.5)&=\tfrac12\ln(4)-\tfrac12\ln(13/3)+\tfrac12\ln(21/5)\\'
             +rf'&\approx {a[0]:.6f}-{a[1]:.6f}+{a[2]:.6f}'
             +rf'={result["predictions"][1]["score"]:.6f}>0,\qquad H(5.5)=+1.'
             +r'\end{align*}'+'\n'
             '四筆資料的加權分數皆不為零，因此預測沒有平手情形。\n')
    (sections / 'predictions.tex').write_text(text)

    text = r'''\section*{七、思考題}
\textbf{Q1. 第一輪分錯的樣本，為何第二輪容易重複抽到？}

ID 8、9 的權重由 0.1 升至 0.25，每筆是分對樣本權重 0.0625 的四倍；若依此機率有放回抽樣，就較容易重複抽到它們。
本例第二輪的固定清單中，ID 8 出現兩次、ID 9 出現三次，但權重較高不保證每次抽樣都會重複。

\textbf{Q2. 為何計算 $\epsilon_2$ 仍要回到原始資料及其權重？}

抽樣資料只用來選規則，$\epsilon_2$ 的定義則是原始 10 筆資料中錯分樣本的權重總和。
本例規則 E 的抽樣錯誤率為 $1/10$，但原始資料錯分 ID 5、6、7，故 $\epsilon_2=3/16$。

\textbf{Q3. 為何第二輪可能選出不同切分規則？}

抽樣組成改變後，各規則的未加權錯誤筆數也隨之改變；第一輪 A 錯 2 筆、E 錯 3 筆，所以選 A。
第二輪重複抽到 ID 8、9，使 A 錯 5 筆，而 E 只錯 1 筆，因此改選切分點 9.5 的 E。
'''
    (sections / 'questions.tex').write_text(text)
