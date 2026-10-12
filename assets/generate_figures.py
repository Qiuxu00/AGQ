"""Regenerate repository result figures from the released aggregate data."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
DATA = ROOT / 'results'
ORANGE, NAVY, TEAL, GRAY = '#C2672E', '#18344E', '#187E83', '#64748B'
LIGHT = '#E7EDF2'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': NAVY, 'axes.labelcolor': NAVY, 'axes.edgecolor': LIGHT,
                     'xtick.color': GRAY, 'ytick.color': GRAY,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.spines.bottom': False,
                     'svg.fonttype': 'path', 'svg.hashsalt': 'agq-results-v1', 'savefig.facecolor': 'white'})


def load(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.18)
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', pad_inches=.18, metadata={'Date': None})
    plt.close(fig)


def title(fig, headline, subtitle):
    fig.text(.055, .96, headline, fontsize=20, fontweight='bold', va='top')
    fig.text(.055, .885, subtitle, fontsize=10, color=GRAY, va='top')


def clean(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=LIGHT, lw=.7)
    ax.tick_params(axis='both', length=0, pad=7)


def overview(q):
    fig = plt.figure(figsize=(11.8, 5.2))
    title(fig, 'Low-bit policies. Measured in closed loop.', 'OpenVLA-OFT on LIBERO  |  600 episodes per configuration  |  4 equally weighted suites')
    ax = fig.add_axes([.07, .21, .52, .57])
    ids = ['w4', 'w3', 'w25', 'w2', 'w15']
    xs = np.arange(5)
    values = [q[c]['success_percent'] for c in ids]
    clean(ax)
    ax.fill_between(xs, values, alpha=.075, color=ORANGE)
    ax.plot(xs, values, color=ORANGE, lw=2.5, marker='o', markersize=7, zorder=4, label='AGQ (measured)')
    for x, y in zip(xs, values):
        ax.text(x, y+3.5, f'{y:.1f}%', ha='center', fontsize=11, fontweight='bold', color=ORANGE)
    qvla = {r[1]: r[-1] for r in load('published_baselines.json')['rows'] if r[0] == 'QVLA'}
    ax.plot([0, 1, 2], [qvla[b] for b in [4, 3, 2.5]], color=GRAY, lw=1.4, ls='--', marker='s', markersize=4,
            label='QVLA (reported by ActQuant)')
    ax.set_xticks(xs, ['W4', 'W3', 'W2.5', 'W2', 'W1.5*'])
    ax.set_ylim(-4, 111)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Task success (%)')
    ax.legend(loc='lower left', bbox_to_anchor=(-.015, -.30), frameon=False, fontsize=9)
    hx = fig.add_axes([.69, .21, .26, .57])
    matrix = np.array([[q[c]['by_suite'][s]['successes']/1.5 for s in ['spatial', 'object', 'goal', 'long']] for c in ids])
    cmap = LinearSegmentedColormap.from_list('agq_teal', ['#F2F5F7', '#C5E1DE', '#187E83'])
    hx.imshow(matrix, vmin=0, vmax=100, cmap=cmap, aspect='auto')
    for i in range(5):
        for j in range(4):
            hx.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', color='white' if matrix[i,j]>73 else NAVY, fontsize=9)
    hx.set_xticks(range(4), ['Spatial', 'Object', 'Goal', 'Long'], fontsize=8.5)
    hx.set_yticks(range(5), ['W4', 'W3', 'W2.5', 'W2', 'W1.5*'])
    hx.tick_params(length=0, pad=7)
    hx.set_title('Suite success (%)', loc='left', fontsize=11, fontweight='bold', pad=10)
    for spine in hx.spines.values():
        spine.set_visible(False)
    fig.text(.055, .015, 'W1.5*: W4 vision + W1.5 language (~1.75 target-weight bits). QVLA values are literature references; unreported points are omitted.',
             color=GRAY, fontsize=8.2)
    save(fig, 'overview')


def lambda_sweep(q):
    fig = plt.figure(figsize=(11.8, 4.9))
    title(fig, 'The allocation matters.', 'Five tested cost weights, one calibrated random-allocation control; every point uses 600 closed-loop episodes.')
    ax = fig.add_axes([.07, .19, .51, .57])
    ids = ['w25_lambda0', 'w25_lambda025', 'w25', 'w25_lambda075', 'w25_lambda1', 'random_w25']
    vals = [q[c]['success_percent'] for c in ids]
    colors = [NAVY, NAVY, ORANGE, NAVY, NAVY, '#A8B4BF']
    clean(ax)
    bars = ax.bar(np.arange(6), vals, width=.64, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+1.6, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(6), ['0', '0.25', '0.5', '0.75', '1', 'Random'])
    ax.set_ylabel('Overall success (%)')
    ax.set_xlabel('Lambda at W2.5')
    ax.set_ylim(0, 85)
    bx = fig.add_axes([.69, .19, .26, .57])
    rows = [r for r in load('allocations.json')['rows'] if r['budget'] == 'W2.5']
    cum = np.zeros(5)
    for index, color in enumerate(['#DBE2E8', '#8DAFBF', TEAL, NAVY]):
        values = np.array([r['combined'][index] for r in rows])
        bx.barh(range(5), values, left=cum, color=color, label=f'W{index+1}', height=.63)
        for y, val, left in zip(range(5), values, cum):
            if val >= 19:
                bx.text(left+val/2, y, str(val), va='center', ha='center', fontsize=8, color='white' if index>1 else NAVY)
        cum += values
    bx.set_yticks(range(5), ['0', '.25', '.5', '.75', '1'])
    bx.set_xlabel('Target matrices (428 total)')
    bx.set_ylabel('Lambda')
    bx.set_xlim(0, 428)
    bx.invert_yaxis()
    bx.tick_params(length=0)
    bx.set_title('The resulting precision mix', loc='left', fontsize=11, fontweight='bold', pad=10)
    bx.legend(ncol=4, loc='lower center', bbox_to_anchor=(.45, -.43), frameon=False, fontsize=8, handlelength=1, columnspacing=.8)
    fig.text(.055, .01, 'Counts describe matrices, not parameter-weighted bits. Lambda=0.5 wins the tested overall average, not every individual suite.', fontsize=8.5, color=GRAY)
    save(fig, 'lambda_sweep')


def resource_tradeoff():
    perf = load('performance.json')
    fig = plt.figure(figsize=(11.8, 5.1))
    title(fig, 'Storage, memory, and latency are separate choices.', 'RTX3090  |  10 fixed-input model-action calls per record  |  8 action steps per call  |  decimal GB')
    ax = fig.add_axes([.07, .23, .44, .54])
    rows = [r for r in perf['rows'] if r['configuration'] in ['bf16', 'w3', 'w25', 'w15']]
    rows = [r for r in rows if r['configuration'] != 'w25' or r['format'] == 'original_packed']
    labels = [('BF16 reference' if r['configuration'] == 'bf16' else r['configuration'].upper().replace('W15', 'W1.5*').replace('W25', 'W2.5'))
              + (' + coding' if r['format']=='grouped_huffman' else '') for r in rows]
    vals = [r['model_bytes']/1e9 for r in rows]
    colors = ['#A8B4BF' if r['configuration']=='bf16' else ORANGE if r['format']=='grouped_huffman' else NAVY for r in rows]
    bars=ax.barh(range(len(rows)), vals, color=colors, height=.57)
    for b, v in zip(bars, vals):
        ax.text(v+.15, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(rows)), labels, fontsize=9)
    ax.set_xlim(0,18)
    ax.invert_yaxis()
    ax.set_xlabel('Complete-model storage (GB)')
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=LIGHT)
    bx=fig.add_axes([.62, .23, .32, .54])
    pts = [r for r in perf['rows'] if r['configuration']=='w15']
    alt = next(r for r in perf['alternative_codec_measurements'] if r['configuration']=='w15' and r['implementation']=='CODEBOOK001')
    bx.scatter([r['latency_ms'] for r in pts]+[alt['latency_ms']], [r['peak_allocated_bytes']/1e9 for r in pts]+[alt['peak_allocated_bytes']/1e9],
               s=[90,100,90], c=[NAVY,ORANGE,TEAL], zorder=3, edgecolors='white', linewidths=1)
    for row, name, offset, color, alignment in zip(
        pts+[alt], ['Original packed','CODEBOOK003','CODEBOOK001'], [(13,16),(14,-23),(-3,15)], [NAVY,ORANGE,TEAL], ['left','left','right']):
        x, y = row['latency_ms'], row['peak_allocated_bytes']/1e9
        bx.annotate(f'{name}\n{y:.3f} GB / {x:.1f} ms', (x,y), xytext=offset, textcoords='offset points', ha=alignment, fontsize=9, color=color)
    bx.set_ylim(2.62,3.9)
    bx.set_xlim(80,1910)
    bx.set_xlabel('Model-action latency (ms)')
    bx.set_ylabel('Peak allocated GPU memory (GB)')
    bx.set_title('W1.5*: three implementation points', loc='left', fontsize=10, fontweight='bold', pad=12)
    clean(bx)
    fig.text(.055,.065, 'Coding improves storage while adding runtime cost. EXP019 quality uses original packed weights; coded trajectory equivalence is not established.', fontsize=8.5,color=GRAY)
    fig.text(.055,.018, 'Peak allocation excludes allocator reserve and non-PyTorch memory. Historical BF16 timing is a reference, not a matched repetition.', fontsize=8.5,color=GRAY)
    save(fig, 'resource_tradeoff')


def task_heatmap(q):
    with (DATA/'tasks.csv').open(encoding='utf-8', newline='') as stream:
        tasks = {(r['configuration'],r['suite'],int(r['task_index'])):int(r['successes']) for r in csv.DictReader(stream)}
    fig, axes = plt.subplots(2,2,figsize=(11.8,6.3))
    fig.subplots_adjust(left=.085,right=.95,top=.77,bottom=.18,wspace=.20,hspace=.55)
    title(fig, 'Every task counts.', 'All 40 tasks, including low-bit failures. Each cell is the number of successes out of 15 episodes.')
    ids=['w4','w3','w25','w2','w15']
    cmap=LinearSegmentedColormap.from_list('agq_heatmap',['#F4F6F8','#BFD9D8',TEAL])
    for ax, suite in zip(axes.flat,['spatial','object','goal','long']):
        values=np.array([[tasks[c,suite,t] for t in range(10)] for c in ids])
        ax.imshow(values,vmin=0,vmax=15,cmap=cmap,aspect='auto')
        for i in range(5):
            for j in range(10):
                ax.text(j,i,str(values[i,j]),ha='center',va='center',fontsize=8.5,color='white' if values[i,j]>11 else NAVY)
        ax.set_title(suite.title(),loc='left',fontsize=12,fontweight='bold')
        ax.set_xticks(range(10),[f'{t:02d}' for t in range(10)],fontsize=8)
        ax.set_yticks(range(5),['W4','W3','W2.5','W2','W1.5*'],fontsize=8)
        ax.tick_params(length=0,pad=5)
    fig.text(.055,.065,'Task indices follow the zero-based LIBERO suite order. W1.5* retains W4 vision and W1.5 language.',fontsize=9,color=GRAY)
    fig.text(.055,.025,'The complete CSV also includes every lambda setting and the calibrated random control; no tasks have been filtered by outcome.',fontsize=9,color=GRAY)
    save(fig,'task_heatmap')


if __name__ == '__main__':
    quality = {r['id']: r for r in load('quality.json')['rows']}
    overview(quality)
    lambda_sweep(quality)
    resource_tradeoff()
    task_heatmap(quality)
    print('Generated four PNG charts and four SVG charts from released results.')
