##
# @file plot.py
# @brief Spiking Neural Networkバイナリ可視化スクリプト
#
# @details
# 本スクリプトは、Spiking Neural Network（SNN）の学習経過を保存したバイナリファイル（ニューロン・シナプス）を読み取り、
# 各種履歴をPNG形式の時系列グラフとして出力します。
#
# @par バイナリファイルの構造
# - ニューロンbin: n_steps(int32), membrane_potential(float32[n]), receptor_pool(float32[n]), input_current(float32[n]), spike(uint8[n])
# - シナプスbin: n_steps(int32), pre_spike(uint8[n]), post_spike(uint8[n]), weight_history(float32[n]), receptor_pool_history(float32[n])
#
# @par コマンドライン引数による使い方
# ```
# python plot.py [result_dir]
# ```
# - `result_dir`: binファイルが保存されたディレクトリを指定（省略時は`./spiking_results`）。
#   例: `python plot.py ./experiment_XYZ`
#
# @author Kazumasa Horie
# @date 2025/8/6 ~
##

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import argparse
import json
import struct

import networkx as nx
import matplotlib.patches as patches
from matplotlib.lines import Line2D

NODE_TYPE = {0: "INP", 1: "SPK", 2: "OUT"}
EDGE_TYPE = {0: "AMPA", 1: "GABA"}

##
# @brief ニューロンのバイナリファイルを読み込む関数（JSONヘッダ対応版）
# @details
# ニューロンデータ（*.bin）は以下のバイナリ構造を持つ：
# - [0:4)        : header_len (int32) — JSONヘッダ文字列のバイト数
# - [4:4+header_len): ヘッダ（UTF-8エンコードJSON文字列, dictとして読み出し可）
# - [4+header_len:4+header_len+4): n_steps (int32) — 記録ステップ数
# - [以降]       : Membrane potential履歴(float32, n_steps分)
# - [以降]       : Receptor pool履歴(float32, n_steps分)
# - [以降]       : Input current履歴(float32, n_steps分)
# - [以降]       : Spike履歴(uint8, n_steps分)（1=発火, 0=非発火）
# 各履歴は時系列で1 stepごとに記録されている。
# @param[in] filename バイナリファイルパス
# @return (ic, mp, spk, rp, n_steps, header) 各種履歴配列・ステップ数・ヘッダdict
##
def load_neuron_bin(filename):
    """
    ニューロンバイナリ（JSONヘッダ付き）の読み込み
    """
    with open(filename, 'rb') as f:
        # --- ヘッダ部分 ---
        header_len_bytes = f.read(4)
        header_len = struct.unpack('i', header_len_bytes)[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)

        # --- 履歴部分 ---
        data = f.read()
        offset = 0
        n_steps = int(np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0])
        offset += 4
        mp = np.frombuffer(data, dtype=np.float32, count=n_steps, offset=offset)
        offset += 4 * n_steps
        rp = np.frombuffer(data, dtype=np.float32, count=n_steps, offset=offset)
        offset += 4 * n_steps
        ic = np.frombuffer(data, dtype=np.float32, count=n_steps, offset=offset)
        offset += 4 * n_steps
        spk = np.frombuffer(data, dtype=np.uint8, count=n_steps, offset=offset)

    return ic, mp, spk, rp, n_steps, header

##
# @brief シナプスのバイナリファイルを読み込む関数
# @details
# シナプスデータ（*.bin）は以下のバイナリ構造を持つ（JSONヘッダ付き）：
# - [0:4)                   : header_len (int32) — JSONヘッダ文字列のバイト数
# - [4 : 4+header_len)       : header_json (UTF-8エンコードJSON文字列)
#
# --- 以下、データ本体 ---
# - [offset: offset+4)       : n_steps (int32) — 記録ステップ数
# - [offset+4 : offset+4+n)  : preニューロンのspike履歴 (uint8, n_steps分)
# - [offset+4+n : offset+4+2n)        : postニューロンのspike履歴 (uint8, n_steps分)
# - [offset+4+2n : offset+4+2n+4n)    : 重み履歴 (float32, n_steps分)
# - [offset+4+2n+4n : offset+4+2n+8n) : レセプタープール履歴 (float32, n_steps分)
#
# ※ 上記の offset は (4 + header_len) に相当。
#    n = n_steps、4n は float32 * n_steps のバイト数(4バイト×n_steps)。
# # spike履歴は 1=発火, 0=非発火。
# @param[in] filename バイナリファイルパス
# @return (pre_spk, post_spk, w_hist, rec_hist, n_steps)
##
def load_synapse_bin(filename):
    with open(filename, 'rb') as f:
        # 1. ヘッダ部：4バイトのjson長＋json本体
        header_len_bytes = f.read(4)
        header_len = struct.unpack('i', header_len_bytes)[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)

        # 2. 履歴データ部（全読み込み＋offset管理）
        data = f.read()
        offset = 0

        n_steps = int(np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0])
        offset += 4
        pre_spk = np.frombuffer(data, dtype=np.uint8, count=n_steps, offset=offset)
        offset += n_steps
        post_spk = np.frombuffer(data, dtype=np.uint8, count=n_steps, offset=offset)
        offset += n_steps
        w_hist = np.frombuffer(data, dtype=np.float32, count=n_steps, offset=offset)
        offset += 4 * n_steps
        rec_hist = np.frombuffer(data, dtype=np.float32, count=n_steps, offset=offset)

    return pre_spk, post_spk, w_hist, rec_hist, n_steps, header

##
# @brief ニューロンのバイナリデータを可視化しPNG保存する
# @details
# ニューロンごとに、入力電流・膜電位・スパイク履歴・レセプタープールの
# 時系列グラフを生成する。
# @param[in] filename バイナリファイル名
# @param[in] save_png PNG保存する場合True、表示のみの場合False
##
def plot_neuron_bin(filename, save_png=True):
    ic, mp, spk, rp, n_steps, header = load_neuron_bin(filename)
    t = np.arange(n_steps)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8),
                             gridspec_kw={'height_ratios':[2,2,1,2]})
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.93, hspace=0.28)

    axes[0].plot(t, ic, color='orange', lw=1.0)
    axes[0].set_ylabel('Input current')
    axes[0].set_facecolor('white')

    axes[1].plot(t, mp, color='b', lw=1.0)
    if "v_th" in header:
        axes[1].axhline(y=header["v_th"], color='red', linestyle='dashed', linewidth=2)
    axes[1].set_ylabel('Membrane potential')
    axes[1].set_facecolor('white')

    axes[2].vlines(t[spk > 0], ymin=0, ymax=1, color='k', lw=0.8)
    axes[2].set_ylim(0, 1.2)
    axes[2].set_ylabel('Spike')
    axes[2].set_yticks([0,1])
    axes[2].set_facecolor('white')

    axes[3].plot(t, rp, color='g', lw=1.0)
    axes[3].set_ylabel('Receptor pool')
    axes[3].set_xlabel('Step')
    axes[3].set_facecolor('white')

    basename = os.path.splitext(os.path.basename(filename))[0]
    fig.suptitle(basename, fontsize=16, y=0.98)

    if save_png:
        out_png = os.path.splitext(filename)[0] + ".png"
        plt.savefig(out_png, dpi=150, facecolor='white')
        plt.close(fig)
        print(f"Saved: {out_png}")
    else:
        plt.show()

##
# @brief シナプスのバイナリデータを可視化しPNG保存する
# @details
# シナプスごとに、pre/postニューロンのspike履歴・重み履歴・レセプタープールの
# 時系列グラフを生成する。
# ファイル名の"AMPA"/"GABA"からタイプを自動判別し、ラベル付けする。
# @param[in] filename バイナリファイル名
# @param[in] save_png PNG保存する場合True、表示のみの場合False
##
def plot_synapse_bin(filename, save_png=True):
    pre_spk, post_spk, w_hist, rec_hist, n_steps, _ = load_synapse_bin(filename)
    t = np.arange(n_steps)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8),
                             gridspec_kw={'height_ratios':[1,1,2,1]})
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.92, hspace=0.25)

    base = os.path.splitext(os.path.basename(filename))[0]
    if 'AMPA' in base:
        syn_type = 'AMPA (Excitatory)'
        weight_color = 'orange'
    elif 'GABA' in base:
        syn_type = 'GABA (Inhibitory)'
        weight_color = 'purple'
    else:
        syn_type = 'Unknown'
        weight_color = 'grey'

    axes[0].vlines(t[pre_spk > 0], ymin=0, ymax=1, color='b', lw=0.7)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel('Pre spike')
    axes[0].set_yticks([0,1])
    axes[0].set_facecolor('white')

    axes[1].vlines(t[post_spk > 0], ymin=0, ymax=1, color='g', lw=0.7)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_ylabel('Post spike')
    axes[1].set_yticks([0,1])
    axes[1].set_facecolor('white')

    axes[2].plot(t, w_hist, color=weight_color, lw=1.2)
    axes[2].set_ylabel('W')
    axes[2].set_facecolor('white')

    axes[3].plot(t, rec_hist, color='teal', lw=1.2)
    axes[3].set_ylabel('Receptor Pool')
    axes[3].set_xlabel('Step')
    axes[3].set_facecolor('white')

    fig.suptitle(f"{base} [{syn_type}]", fontsize=15, y=0.99)

    if save_png:
        out_png = os.path.splitext(filename)[0] + ".png"
        plt.savefig(out_png, dpi=150, facecolor='white')
        plt.close(fig)
        print(f"Saved: {out_png}")
    else:
        plt.show()


def load_snn_bin(filename):
    nodes = []
    edges = []
    with open(filename, "rb") as f:
        N, = struct.unpack("i", f.read(4))
        for _ in range(N):
            layer, nid = struct.unpack("ii", f.read(8))
            t, = struct.unpack("B", f.read(1))
            nodes.append((layer, nid, t))
        M, = struct.unpack("i", f.read(4))
        for _ in range(M):
            pre_l, pre_id = struct.unpack("ii", f.read(8))
            post_l, post_id = struct.unpack("ii", f.read(8))
            t, = struct.unpack("B", f.read(1))
            edges.append((pre_l, pre_id, post_l, post_id, t))
    return nodes, edges

def build_nx_graph(nodes, edges):
    G = nx.MultiDiGraph()
    for (layer, nid, t) in nodes:
        G.add_node((layer, nid),
                   label=f"{layer}_{nid}",
                   type=NODE_TYPE.get(t, "UNK"))
    for (pre_l, pre_id, post_l, post_id, t) in edges:
        G.add_edge((pre_l, pre_id), (post_l, post_id),
                   type=EDGE_TYPE.get(t, "UNK"))
    return G

def layered_layout(G):
    pos = {}
    layers = {}
    for n in G.nodes:
        layer = n[0]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(n)
    for l, nodes in layers.items():
        for i, node in enumerate(sorted(nodes, key=lambda x: x[1])):  # neuron_id順
            pos[node] = (l, -i)  # X=layer, Y=neuron index（下から上）
    return pos

def merge_edge_types(edges):
    # (pre, post)ごとにセット
    edge_map = {}
    for pre_l, pre_id, post_l, post_id, t in edges:
        key = ((pre_l, pre_id), (post_l, post_id))
        if key not in edge_map:
            edge_map[key] = set()
        edge_map[key].add(t)
    return edge_map

def draw_graph(G, edges, out_png="network_graph.png"):
    pos = layered_layout(G)
    node_colors = []
    for n in G.nodes:
        t = G.nodes[n]["type"]
        if t == "INP":
            node_colors.append("skyblue")
        elif t == "OUT":
            node_colors.append("orange")
        else:
            node_colors.append("lightgreen")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes}, font_size=10)
    
    # --- ここでAMPAとGABAの重複を1本に統合 ---
    edge_map = merge_edge_types(edges)
    for (u, v), types in edge_map.items():
        if 0 in types and 1 in types:    # 両方あり
            color = "mediumpurple"
            width = 3.0
        elif 1 in types:                 # GABAのみ
            color = "royalblue"
            width = 2.0
        else:                            # AMPAのみ
            color = "indianred"
            width = 2.0
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=color,
            width=width,
            arrowstyle='-|>',
            arrowsize=18,
            connectionstyle='arc3,rad=0.0'
        )
    # 凡例
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Input', markerfacecolor='skyblue', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Spiking', markerfacecolor='lightgreen', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Output', markerfacecolor='orange', markersize=12),
        Line2D([0], [0], color='indianred', lw=2, label='AMPA'),
        Line2D([0], [0], color='royalblue', lw=2, label='GABA'),
        Line2D([0], [0], color='mediumpurple', lw=3, label='AMPA+GABA')
    ]
    plt.legend(handles=legend_elems, loc='best')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")
    
def extract_subgraph(nodes, edges, center_layer, center_id, in_max=10, out_max=10):
    """
    指定ノードを中心に、「接続元ノード数が最大in_max」「接続先ノード数が最大out_max」まで拾い、
    その間のエッジは**全種類**含める。
    """
    center = (center_layer, center_id)

    # 入力側：uniqueな接続元ノードをin_max個ピックアップ
    in_edges_all = [e for e in edges if (e[2], e[3]) == center]
    # (pre_layer, pre_id)ごとにまとめる
    in_node_dict = {}
    for e in in_edges_all:
        pre_node = (e[0], e[1])
        if pre_node not in in_node_dict:
            in_node_dict[pre_node] = []
        in_node_dict[pre_node].append(e)
    # 最大in_maxノード分だけ残す
    in_nodes = list(in_node_dict.keys())[:in_max]
    # それらから来ているエッジは**全て**残す
    in_edges = []
    for pre_node in in_nodes:
        in_edges.extend(in_node_dict[pre_node])

    # 出力側も同様
    out_edges_all = [e for e in edges if (e[0], e[1]) == center]
    out_node_dict = {}
    for e in out_edges_all:
        post_node = (e[2], e[3])
        if post_node not in out_node_dict:
            out_node_dict[post_node] = []
        out_node_dict[post_node].append(e)
    out_nodes = list(out_node_dict.keys())[:out_max]
    out_edges = []
    for post_node in out_nodes:
        out_edges.extend(out_node_dict[post_node])

    # 表示ノードは中心＋入力＋出力ノード
    sub_node_set = set(in_nodes + out_nodes + [center])
    sub_nodes = [n for n in nodes if (n[0], n[1]) in sub_node_set]
    # 表示エッジは上記で拾ったもの
    sub_edges = in_edges + out_edges

    return sub_nodes, sub_edges

def plot_snn_layers_box_vertical(
    layer_types,
    edge_types,
    neuron_stats=None,
    out_png="snn_layers_box.png",
    layer_names=None,
    layer_colors=None,
    edge_colors=None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D

    N = len(layer_types)
    ys = list(reversed(range(N)))
    hs = [0.2] * N
    base_w = 2.0  # 横幅
    # 色
    if layer_colors is None:
        layer_colors = ["skyblue", "lightgreen", "orange", "plum", "lightgray"] * 3
    if edge_colors is None:
        edge_colors = {"AMPA":"firebrick", "GABA":"steelblue", "BOTH":"slateblue"}
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in ys]

    # 横幅（最大ニューロン数で正規化しやすくする例）
    max_count = max(sum(d.values()) for d in layer_types)
    ws = [max(1.8, base_w * (sum(d.values())/max_count)**0.9) for d in layer_types]

    fig, ax = plt.subplots(figsize=(7, 1.5*N+2))
    x_center = 3.2  # 水平中心
    # 層ボックス（横長でラベルも入れやすい）
    for i, (y, h, w, d, label) in enumerate(zip(ys, hs, ws, layer_types, layer_names)):
        x = x_center - w/2
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.09",
            fc=layer_colors[i%len(layer_colors)], ec="k", lw=2
        )
        ax.add_patch(rect)
        txt = "  ".join([f"{k}: {v}" for k,v in d.items()])
        ax.text(x_center, y + h/2, f"{label}   {txt}",
                ha="center", va="center", fontsize=15, fontweight="bold")
        if neuron_stats and neuron_stats[i]:
            ax.text(x_center, y + h - 0.05, neuron_stats[i], ha="center", va="bottom", fontsize=12, color="#444")


    # 層間エッジ（縦方向中心どうし）
    for i, (y1, h1, w1, y2, h2, w2, d) in enumerate(zip(ys[:-1], hs[:-1], ws[:-1], ys[1:], hs[1:], ws[1:], edge_types)):
        x1 = x_center
        x2 = x_center
        # 始点は上側ボックスの下端
        start_y = y1
        # 終点は下側ボックスの上端
        end_y = y2+h2
        # 色・太さ
        color = None
        if "AMPA" in d and "GABA" in d:
            if d["AMPA"] > 0 and d["GABA"] > 0:
                color = edge_colors["BOTH"]
            elif d["AMPA"] > 0:
                color = edge_colors["AMPA"]
            elif d["GABA"] > 0:
                color = edge_colors["GABA"]
        elif "AMPA" in d and d["AMPA"] > 0:
            color = edge_colors["AMPA"]
        elif "GABA" in d and d["GABA"] > 0:
            color = edge_colors["GABA"]
        else:
            color = "gray"
        total = sum(d.values())
        width = max(2, min(5, (total / max_count) * 7.5))
        ax.annotate("",
            xy=(x2, end_y), xycoords='data',
            xytext=(x1, start_y), textcoords='data',
            arrowprops=dict(arrowstyle="->", lw=width, color=color, shrinkA=0, shrinkB=0),
        )
        # ラベルは矢印の右横
        ax.text(x2 + w2/2 + 0.16, (start_y + end_y)/2,
                "  ".join([f"{k}:{v}" for k,v in d.items() if v>0]),
                ha="left", va="center", fontsize=13, color=color, fontweight="bold")
        
    ax.set_ylim(-0.5, N+0.6)
    ax.set_xlim(0.3, 6.3)
    ax.axis("off")

    # 凡例（下部中央）
    legend_elems = [
        Line2D([0], [0], color=edge_colors["AMPA"], lw=3, label='AMPA'),
        Line2D([0], [0], color=edge_colors["GABA"], lw=3, label='GABA'),
        Line2D([0], [0], color=edge_colors["BOTH"], lw=4, label='AMPA+GABA'),
    ]
    ax.legend(handles=legend_elems, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")

def summarize_layers_and_edges_from_bin(bin_file):
    """
    network_graph.binから
    - 各層のニューロン数（種類別：INP/SPK/OUT）
    - 各層間のエッジ本数（種類別：AMPA/GABA/AMPA+GABA）
    を自動集計してplot_snn_layers_boxへ渡すラッパー
    """
    nodes, edges = load_snn_bin(bin_file)

    # 1. 層数推定（最大layer+1）
    layer_ids = [n[0] for n in nodes]
    num_layers = max(layer_ids) + 1

    # 2. 各層ごとにニューロン種類別カウント
    layer_types = []
    for l in range(num_layers):
        d = {"INP": 0, "SPK": 0, "OUT": 0}
        for n in nodes:
            if n[0] == l:
                d[n[2] if isinstance(n[2], str) else NODE_TYPE.get(n[2], "UNK")] += 1
        # 0しかいない種類は省略してもOK
        d = {k: v for k, v in d.items() if v > 0}
        layer_types.append(d)

    # 3. 各層間エッジの種類別カウント
    # edge: (pre_l, pre_id, post_l, post_id, t) ← t:0=AMPA, 1=GABA
    edge_types = []
    for l in range(num_layers - 1):
        d = {"AMPA": 0, "GABA": 0}
        for e in edges:
            if e[0] == l and e[2] == l+1:
                d[EDGE_TYPE.get(e[4], "UNK")] += 1
        # 0しかいない種類は省略してもOK
        d = {k: v for k, v in d.items() if v > 0}
        edge_types.append(d)

    # 4. 各層の合計スパイクなど（省略 or 拡張可）
    neuron_stats = [None for _ in range(num_layers)]  # 必要なら追加

    # 5. 層名
    layer_names = []
    for i, lt in enumerate(layer_types):
        if "INP" in lt and len(lt) == 1:
            name = "Input"
        elif "OUT" in lt and len(lt) == 1:
            name = "Output"
        else:
            name = "Hidden"
        layer_names.append(name)

    return layer_types, edge_types, neuron_stats, layer_names

# ↑これをそのまま使って
def plot_snn_box_from_bin(bin_file, out_png="snn_layers_box.png"):
    layer_types, edge_types, neuron_stats, layer_names = summarize_layers_and_edges_from_bin(bin_file)
    plot_snn_layers_box_vertical(
        layer_types, edge_types,
        neuron_stats=neuron_stats,
        out_png=out_png,
        layer_names=layer_names
    )

##
# @brief ディレクトリ内の全バイナリデータ（ニューロン/シナプス）を自動プロット
# @details
# result_dir配下のbinファイルを正規表現で分類し、対応するplot関数でPNG化。
# エラー時はファイル名とエラー内容をprint出力。
# コマンドライン引数でresult_dirを切り替え可能。
##
if __name__ == "__main__":
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(
        description="Spiking Neural Networkのバイナリ可視化スクリプト")
    parser.add_argument(
        "result_dir", nargs="?", default="./spiking_results",
        help="binファイルの保存ディレクトリ（省略時は./spiking_results）"
    )
    args = parser.parse_args()
    result_dir = args.result_dir

    neuron_bin_pattern = re.compile(r'^\d+_\d+_[A-Z]{3}_(\d{4}|final)\.bin$')
    synapse_bin_pattern = re.compile(r'^syn_\d+_\d+_to_\d+_\d+_(AMPA|GABA)_\d{4}\.bin$')    

    all_bin_files = sorted(glob.glob(os.path.join(result_dir, "*.bin")))
    
    neuron_bin_files = [f for f in all_bin_files if neuron_bin_pattern.match(os.path.basename(f))]
    synapse_bin_files = [f for f in all_bin_files if synapse_bin_pattern.match(os.path.basename(f))]

    print(f"Found {len(neuron_bin_files)} neuron bin files.")
    print(f"Found {len(synapse_bin_files)} synapse bin files.")

    for bin_file in neuron_bin_files:
        try:
            plot_neuron_bin(bin_file, save_png=True)
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")

    for bin_file in synapse_bin_files:
        try:
            plot_synapse_bin(bin_file, save_png=True)
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")

    # --- Network構造グラフ(bin→png)自動化 ---
    graph_bin = os.path.join(result_dir, "network_graph.bin")
    center_layer = 1
    center_id = 0

    if os.path.exists(graph_bin):
        try:
            nodes, edges = load_snn_bin(graph_bin)
            out_png = os.path.join(result_dir, "network_graph_"+str(center_layer)+"_"+str(center_id)+".png")
            sub_nodes, sub_edges = extract_subgraph(nodes, edges, center_layer, center_id, in_max=10, out_max=10)
            G = build_nx_graph(sub_nodes, sub_edges)
            draw_graph(G, sub_edges, out_png)
        except Exception as e:
            print(f"Error processing {graph_bin}: {e}")
        plot_snn_box_from_bin(graph_bin, os.path.join(result_dir, "snn_layers_box.png"))
    else:
        print(f"network_graph.bin not found in {result_dir}")

        nodes, edges = load_snn_bin("network_graph.bin")
        