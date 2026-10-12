"""Generate compact case figures from read-only released aggregate data.

Run with the ``figures`` dependencies installed. No model, GPU, or new
experiment is required. Literature rounding and missing values are preserved.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA = ROOT / 'assets', ROOT / 'results'
ORANGE, NAVY, TEAL, GRAY = '#B65C30', '#18344E', '#217F83', '#627080'
LIGHT, PALE, WARM = '#DCE4E9', '#F2F6F8', '#FCF0E7'
SUITES = ['spatial', 'object', 'goal', 'long']
MAIN_IDS = ['w4', 'w3', 'w25', 'w2', 'w15']
LABELS = ['W4', 'W3', 'W2.5', 'W2', 'W1.5*']
CMAP = LinearSegmentedColormap.from_list('agq_teal', ['#F4F7F8', '#BEDCD9', '#176F75'])
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.3,
    'text.color': NAVY, 'axes.labelcolor': NAVY, 'axes.edgecolor': LIGHT,
    'xtick.color': GRAY, 'ytick.color': NAVY,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'svg.fonttype': 'path', 'svg.hashsalt': 'agq-results-v2',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def load(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=200)
    fig.savefig(OUT / f'{name}.svg', metadata={'Date': None})
    plt.close(fig)


def header(fig, headline, subtitle):
    fig.text(.037, .965, headline, fontsize=16, fontweight='bold', va='top')
    fig.text(.037, .905, subtitle, fontsize=9.1, color=GRAY, va='top')
    fig.add_artist(plt.Line2D([.037, .967], [.858, .858], transform=fig.transFigure,
                             color=LIGHT, lw=.8))


def panel(fig, x, y, text):
    fig.text(x, y, text, fontsize=10.7, fontweight='bold', va='top')


def table(fig, bounds):
    ax = fig.add_axes(bounds)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return ax


def rule(ax, y, lw=.7):
    ax.plot([0, 1], [y, y], color=LIGHT, lw=lw, clip_on=False)


def overview(q):
    fig = plt.figure(figsize=(12.8, 4.4))
    header(fig, 'Closed-loop quality across five precision budgets',
           'OpenVLA-OFT / LIBERO  ·  600 episodes per configuration  ·  150 episodes per suite  ·  EXP019')
    panel(fig, .037, .823, 'A  Overall success')
    panel(fig, .535, .823, 'B  Four-suite breakdown: success rate and count')
    ax = fig.add_axes([.09, .225, .405, .515])
    ys, vals = np.arange(5), [q[c]['success_percent'] for c in MAIN_IDS]
    ax.barh(ys, 100, color=PALE, height=.57, zorder=1)
    ax.barh(ys, vals, color=[NAVY, NAVY, ORANGE, NAVY, NAVY], height=.57, zorder=2)
    for y, val, c in zip(ys, vals, MAIN_IDS):
        ax.text(val-1.6, y, f'{val:.1f}%', ha='right', va='center', color='white',
                fontweight='bold', fontsize=10.1)
        ax.text(104, y, f"{q[c]['successes']}/600", va='center', fontsize=9, color=GRAY)
    qvla = {r[1]: r[-1] for r in load('published_baselines.json')['rows'] if r[0] == 'QVLA'}
    ax.scatter([qvla[b] for b in [4, 3, 2.5]], [.34, 1.34, 2.34], marker='D', s=38,
               facecolor='white', edgecolor=GRAY, lw=1.4, zorder=5)
    ax.set_yticks(ys, LABELS, fontsize=10, fontweight='bold')
    ax.set_ylim(4.55, -.55)
    ax.set_xlim(-1, 124)
    ax.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '100%'])
    ax.tick_params(length=0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=LIGHT, lw=.6)
    fig.text(.037, .133, '◇ QVLA (ActQuant): W4 96.8%  /  W3 37.4%  /  W2.5 0.0%', fontsize=9, color=GRAY)
    tx = table(fig, [.535, .222, .431, .551])
    col_x = [.13, .375, .625, .875]
    for x, suite in zip(col_x, SUITES):
        tx.text(x, .94, suite.title(), va='center', ha='center', fontweight='bold')
    for row, c in enumerate(MAIN_IDS):
        y = .775-row*.174
        for x, suite in zip(col_x, SUITES):
            r = q[c]['by_suite'][suite]
            rate = r['successes']/r['episodes']*100
            tx.add_patch(Rectangle((x-.12, y-.078), .24, .158,
                                   facecolor=CMAP(rate/100), edgecolor='white', lw=1))
            ink = 'white' if rate >= 76 else NAVY
            tx.text(x, y+.023, f'{rate:.1f}%', ha='center', va='center',
                    fontsize=10.8, fontweight='bold', color=ink)
            tx.text(x, y-.038, f"{r['successes']}/150", ha='center', va='center', fontsize=8.3, color=ink)
    fig.text(.535, .133, 'Rows follow the same W4 → W1.5* order as panel A.', fontsize=9, color=GRAY)
    fig.text(.037, .074, 'W1.5*: W4 vision + W1.5 language (about 1.75 target-weight bits). QVLA values are literature references, not local runs.', fontsize=8.7, color=GRAY)
    fig.text(.037, .033, 'Unreported QVLA points are omitted. The complete release contains 10 configurations and 6,000 episodes; this panel shows five configurations.', fontsize=8.7, color=GRAY)
    save(fig, 'overview')


def lambda_sweep(q):
    ids = ['w25_lambda0', 'w25_lambda025', 'w25', 'w25_lambda075', 'w25_lambda1', 'random_w25']
    fig = plt.figure(figsize=(12.8, 5.15))
    header(fig, 'Allocation trade-offs at W2.5',
           'Five cost weights + a calibrated random-allocation control  ·  600 closed-loop episodes per row')
    panel(fig, .037, .827, 'A  Overall and suite-level success (%)')
    ax = table(fig, [.037, .19, .555, .586])
    for text, x in [('λ', .035), ('Overall / successes', .275), ('Spatial', .535), ('Object', .67), ('Goal', .805), ('Long', .94)]:
        ax.text(x, .95, text, ha='center', va='center', fontsize=8.8, fontweight='bold')
    rule(ax, .898)
    for i, (c, label) in enumerate(zip(ids, ['0', '0.25', '0.5', '0.75', '1', 'Random'])):
        y = .815-i*.139
        if c == 'w25':
            ax.add_patch(Rectangle((0, y-.065), 1, .13, facecolor=WARM, zorder=0))
        color = ORANGE if c == 'w25' else GRAY if c == 'random_w25' else NAVY
        ax.text(.035, y, label, ha='center', va='center', fontsize=9, color=color, fontweight='bold' if c == 'w25' else 'normal')
        ax.add_patch(Rectangle((.10, y-.017), .175, .034, facecolor=LIGHT))
        ax.add_patch(Rectangle((.10, y-.017), .175*q[c]['success_percent']/100, .034, facecolor=color))
        ax.text(.30, y+.023, f"{q[c]['success_percent']:.1f}%", va='center', fontsize=10, color=color, fontweight='bold')
        ax.text(.30, y-.038, f"{q[c]['successes']}/600", va='center', fontsize=7.8, color=GRAY)
        for x, suite in zip([.535, .67, .805, .94], SUITES):
            r = q[c]['by_suite'][suite]
            ax.text(x, y, f"{100*r['successes']/r['episodes']:.1f}", ha='center', va='center', fontsize=9.5, fontweight='bold' if c == 'w25' else 'normal')
        rule(ax, y-.07, lw=.45)
    panel(fig, .637, .827, 'B  Precision allocation: 428 target matrices')
    bx = fig.add_axes([.675, .438, .291, .302])
    rows = sorted([r for r in load('allocations.json')['rows'] if r['budget'] == 'W2.5'], key=lambda r: r['lambda'])
    cumulative = np.zeros(len(rows))
    for index, color in enumerate(['#D6E3E8', '#8AB5BF', TEAL, NAVY]):
        values = np.array([r['combined'][index] for r in rows])
        bx.barh(range(5), values, left=cumulative, color=color, label=f'W{index+1}', height=.66, edgecolor='white', lw=.6)
        for y, val, left in zip(range(5), values, cumulative):
            bx.text(left+val/2, y, str(val), va='center', ha='center', fontsize=8.2, color='white' if index > 1 else NAVY, fontweight='bold' if y == 2 else 'normal')
        cumulative += values
    bx.set_yticks(range(5), ['λ 0', 'λ .25', 'λ .5', 'λ .75', 'λ 1'], fontsize=9)
    bx.set_xticks([])
    bx.set_xlim(0, 428)
    bx.set_ylim(4.65, -.65)
    bx.tick_params(length=0, pad=6)
    bx.legend(ncol=4, loc='lower center', bbox_to_anchor=(.49, -.15), frameon=False, fontsize=8.5, handlelength=1.1, columnspacing=1.2)
    cx = table(fig, [.637, .17, .33, .185])
    default = next(r for r in rows if r['lambda'] == .5)
    cx.text(0, .93, 'Default λ = 0.5: vision / language split', fontsize=9.2, fontweight='bold')
    for x, bit in zip([.40, .57, .74, .91], [1, 2, 3, 4]):
        cx.text(x, .67, f'W{bit}', ha='center', va='center', fontsize=8.5, color=GRAY)
    for y, key, label in [(.40, 'vision', 'Vision · 204'), (.09, 'language', 'Language · 224')]:
        cx.text(0, y, label, va='center', fontsize=9)
        for x, value in zip([.40, .57, .74, .91], default[key]):
            cx.text(x, y, str(value), va='center', ha='center', fontsize=9.2, fontweight='bold')
    fig.text(.037, .105, 'λ = 0.5 has the highest overall success among the tested settings; the best setting varies by suite.', fontsize=9, color=GRAY)
    fig.text(.037, .053, 'Allocation cells count matrices, not parameters or average bits. Random denotes the calibrated W2.5 control, not an uncalibrated model.', fontsize=8.8, color=GRAY)
    save(fig, 'lambda_sweep')


def resource_tradeoff():
    perf = load('performance.json')
    fig = plt.figure(figsize=(12.8, 6.2))
    header(fig, 'Deployment resources: storage, peak memory, and latency',
           'RTX3090  ·  10 fixed-input model-action calls per record  ·  8 action steps per call  ·  decimal GB')
    panel(fig, .037, .829, 'A  Complete performance table')
    ax = table(fig, [.037, .347, .93, .446])
    for label, x in [('Configuration', .004), ('Weight format', .126), ('Model storage / GB', .35), ('Compression', .52), ('Peak allocated / GB', .71), ('Latency / ms', .918)]:
        ax.text(x, .952, label, va='center', ha='left' if x < .26 else 'center', fontsize=9, fontweight='bold')
    rule(ax, .90)
    labels = {'bf16': 'BF16 †', 'w4': 'W4', 'w3': 'W3', 'w25': 'W2.5', 'w2': 'W2', 'w15': 'W1.5*'}
    for i, r in enumerate(perf['rows']):
        y = .84-i*.107
        coded = r['format'] == 'grouped_huffman'
        color = ORANGE if coded else GRAY if r['configuration'] == 'bf16' else NAVY
        if coded or i % 2 == 0:
            ax.add_patch(Rectangle((0, y-.05), 1, .101, color=WARM if coded else PALE, zorder=0))
        ax.text(.004, y, labels[r['configuration']], va='center', fontweight='bold', color=color, fontsize=9.5)
        ax.text(.126, y, 'CODEBOOK003' if coded else 'BF16' if r['configuration']=='bf16' else 'Packed', va='center', color=color, fontsize=9)
        storage, peak = r['model_bytes']/1e9, r['peak_allocated_bytes']/1e9
        for start, value, ink in [(.265, storage, color), (.625, peak, TEAL)]:
            ax.add_patch(Rectangle((start, y-.016), .097, .032, color=LIGHT))
            ax.add_patch(Rectangle((start, y-.016), .097*value/16.1, .032, color=ink))
            ax.text(start+.115, y, f'{value:.3f}', va='center', fontsize=9.6, fontweight='bold')
        ax.text(.52, y, f"{r['compression_ratio']:.2f}×", va='center', ha='center', fontsize=9.5)
        ax.text(.946, y, f"{r['latency_ms']:.1f}", va='center', ha='right', fontsize=9.5, fontweight='bold' if coded else 'normal', color=color)
        rule(ax, y-.052, lw=.4)
    panel(fig, .037, .321, 'B  Alternative codec measurements: compare each memory / latency pair together')
    ax = table(fig, [.037, .115, .93, .167])
    for left, configuration, label in [(0, 'w2', 'W2'), (.515, 'w15', 'W1.5*')]:
        ax.add_patch(Rectangle((left, .02), .485, .95, facecolor=PALE, edgecolor=LIGHT, lw=.7))
        ax.text(left+.018, .785, label, fontsize=10, fontweight='bold', va='center')
        ax.text(left+.278, .785, 'Peak / GB', fontsize=8.7, ha='center', va='center', color=GRAY)
        ax.text(left+.435, .785, 'Latency / ms', fontsize=8.7, ha='center', va='center', color=GRAY)
        rows = [r for r in perf['alternative_codec_measurements'] if r['configuration'] == configuration]
        for y, r in zip([.46, .16], rows):
            color = TEAL if r['implementation'] == 'CODEBOOK001' else ORANGE
            ax.text(left+.018, y, r['implementation'], fontsize=9.1, va='center', color=color, fontweight='bold')
            ax.text(left+.278, y, f"{r['peak_allocated_bytes']/1e9:.3f}", ha='center', va='center', fontsize=10.2, color=color)
            ax.text(left+.435, y, f"{r['latency_ms']:.1f}", ha='center', va='center', fontsize=10.2, color=color)
    fig.text(.037, .075, '† BF16 is a historical reference. Peak allocation excludes allocator reserve and non-PyTorch memory; timing excludes action pre/postprocessing.', fontsize=8.4, color=GRAY)
    fig.text(.037, .035, 'Quality and resource measurements are separate. Coded weights were not re-evaluated in EXP019; strict coded closed-loop equivalence is not established.', fontsize=8.4, color=GRAY)
    save(fig, 'resource_tradeoff')


def task_heatmap(q):
    with (DATA/'tasks.csv').open(encoding='utf-8', newline='') as stream:
        tasks = {(r['configuration'], r['suite'], int(r['task_index'])): int(r['successes']) for r in csv.DictReader(stream)}
    ids = ['w4', 'w3', 'w25_lambda0', 'w25_lambda025', 'w25', 'w25_lambda075', 'w25_lambda1', 'w2', 'w15', 'random_w25']
    labels = ['W4', 'W3', 'W2.5 · λ0', 'W2.5 · λ.25', 'W2.5 · λ.5', 'W2.5 · λ.75', 'W2.5 · λ1', 'W2', 'W1.5*', 'Random W2.5']
    fig = plt.figure(figsize=(12.8, 7.8))
    fig.text(.037, .972, 'All tasks, all configurations', fontsize=16, fontweight='bold', va='top')
    fig.text(.037, .932, '400 task records  ·  6,000 closed-loop episodes  ·  each cell = successes / 15  ·  right column = suite success (%)', fontsize=9.2, color=GRAY, va='top')
    for panel_index, suite in enumerate(SUITES):
        left = .125 if panel_index % 2 == 0 else .62
        bottom = .54 if panel_index < 2 else .15
        ax = fig.add_axes([left, bottom, .307, .302])
        values = np.array([[tasks[c, suite, t] for t in range(10)] for c in ids])
        ax.imshow(values, vmin=0, vmax=15, cmap=CMAP, aspect='auto', interpolation='nearest')
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(values[i, j]), ha='center', va='center', fontsize=8.6, color='white' if values[i, j] >= 12 else NAVY)
        ax.set_xticks(range(10), [f'{t:02d}' for t in range(10)], fontsize=8.4)
        ax.set_yticks(range(10), labels, fontsize=8.4)
        ax.tick_params(length=0, pad=5)
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
        ax.grid(which='minor', color='white', lw=1.1)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.get_yticklabels()[4].set_color(ORANGE)
        ax.get_yticklabels()[4].set_fontweight('bold')
        ax.set_title(f'{chr(65+panel_index)}  {suite.title()}', loc='left', fontsize=11.3, fontweight='bold', pad=16)
        ax.text(.5, 1.02, 'Task index', transform=ax.transAxes, ha='center', va='bottom', fontsize=8.5, color=GRAY)
        for i, c in enumerate(ids):
            rate = 100*q[c]['by_suite'][suite]['successes']/150
            ax.text(10.02, i, f'{rate:.1f}', ha='left', va='center', fontsize=8.5, color=ORANGE if c == 'w25' else NAVY, fontweight='bold' if c == 'w25' else 'normal', clip_on=False)
        ax.text(10.35, -1.0, 'Suite %', ha='center', va='center', fontsize=8.3, fontweight='bold', clip_on=False)
    legend = fig.add_axes([.727, .070, .23, .017])
    legend.imshow(np.linspace(0, 15, 256).reshape(1, -1), aspect='auto', cmap=CMAP, extent=[0, 15, 0, 1])
    legend.set_yticks([])
    legend.set_xticks([0, 5, 10, 15], ['0', '5', '10', '15'], fontsize=8)
    legend.tick_params(length=0, pad=3)
    fig.text(.037, .085, 'Task indices use the zero-based LIBERO suite order; no task outcomes are filtered.', fontsize=8.8, color=GRAY)
    fig.text(.037, .050, 'W1.5*: W4 vision + W1.5 language. Orange row label: default W2.5 allocation (λ = 0.5).', fontsize=8.8, color=GRAY)
    save(fig, 'task_heatmap')


if __name__ == '__main__':
    quality = {r['id']: r for r in load('quality.json')['rows']}
    overview(quality)
    lambda_sweep(quality)
    resource_tradeoff()
    task_heatmap(quality)
    print('Generated four PNG charts and four SVG charts from released results.')
