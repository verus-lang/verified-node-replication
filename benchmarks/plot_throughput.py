#!/usr/bin/env python3

from plotnine.themes.theme_gray import theme_gray
from plotnine.themes.theme import theme
from plotnine.themes.elements import (element_line, element_rect,
                                      element_text, element_blank)
import sys
import re
import pandas as pd
import numpy as np
import plotnine as p9

from plotnine import *
from plotnine.data import *

import warnings

from io import BytesIO

# this is the width of a column in the latex template
LATEX_TEMPLATE_COLUMNWIDTH =  4# 3.25 # 2.8


# the unit of the latex template column width
LATEX_TEMPLATE_COLUMNWDITH_UNIT = 'in'

# this is the width of the plot
PLOT_WIDTH = LATEX_TEMPLATE_COLUMNWIDTH

# this is the size unit
PLOT_SIZE_UNIT = LATEX_TEMPLATE_COLUMNWDITH_UNIT

# this is the ration of the plot
PLOT_ASPECT_RATIO = 16/7


# this is the plot height
PLOT_HEIGHT = PLOT_WIDTH/PLOT_ASPECT_RATIO

# active categories
CATEGORIES=['verus_nr', 'upstream_nr', 'dafny_nr', 'rust_nr']
# CATEGORIES=['upstream_nr', 'dafny_nr', 'rust_nr', 'verus_nr']
CATEGORIES=[ 'rust_nr',  'dafny_nr',  'verus_nr']

ALL_LABELS={
    'verus_nr': 'Verus-NR',
    'upstream_nr':  'Upstream-NR',
    'dafny_nr':  'IronSync-NR',
    'rust_nr':  'NR',
    'dafny_rwlock':  'DistRwLock',
    'shfllock': 'Shuffle Lock',
    'mcs': 'MCS',
    'cpp_shared_mutex': 'libstdc++ shared_mutex'
}

ALL_COLORS={
    'verus_nr'          : "#d53e4f",
    'upstream_nr'       : "#bababadd",
    'dafny_nr'          : "#1a1a1a",
    'rust_nr'           : "#bababadd",
    'dafny_rwlock'      : "#ff7f00",
    'shfllock'          : "#f781bf",
    'mcs'               : "#999999",
    'cpp_shared_mutex'  : "#a6cee3"
}


ALL_MARKERS={
    'verus_nr': 'D',
    'upstream_nr':  'o',
    'dafny_nr':   'x',
    'rust_nr': '^',
    'dafny_rwlock':  'v',
    'shfllock': '*',
    'mcs': '+',
    'cpp_shared_mutex': 'x'
}


def count_numa_nodes():
    nids = set()
    with open('/proc/cpuinfo') as f:
        for l in f.readlines():
            if l.startswith('physical id'):
                nid = l.split(':')[1].strip()
                nids.add(nid)
    return len(nids)

def count_cores_per_numa_node():
    cores = 0
    with open('/proc/cpuinfo') as f:
        for l in f.readlines():
            if re.match('physical id.*:.0$', l):
                cores += 1
    return cores

NUMA_NODES = count_numa_nodes()
CORES_PER_NODE = count_cores_per_numa_node()

if NUMA_NODES == 4:
    REPLICA_CATEGORIES=[1, 2, 4]
elif NUMA_NODES == 2:
    REPLICA_CATEGORIES=[1, 2]
elif NUMA_NODES == 1:
    REPLICA_CATEGORIES=[1]
else:
    print("Unknown number of NUMA nodes")
    sys.exit(1  )

LABELS = [ALL_LABELS[d] for d in CATEGORIES]
BREAKS=[ d for d in CATEGORIES]
COLORS=[ ALL_COLORS[i] for i in CATEGORIES ]
MARKERS=[ ALL_MARKERS[i] for i in CATEGORIES ]


# What machine, max cores, sockets, revision
MACHINE = ('nr-results', CORES_PER_NODE * NUMA_NODES, NUMA_NODES, '0')



class theme_my538(theme_gray):
    def __init__(self, base_size=7, base_family='Computer Modern Roman'):
        theme_gray.__init__(self, base_size)
        bgcolor = '#FFFFFF'
        self.add_theme(
            theme(
                # title setup
                title=element_text(color='#3C3C3C'),
                # legend setup
                legend_box_margin = 0,
                legend_margin     = 0,
                legend_text       = element_text(size=base_size-1),
                legend_background = element_rect(fill='None', color='#000000', size=0.1, linetype='none'),
                legend_key        = element_rect(fill='#FFFFFF', colour=None),
                # axis setup
                axis_title        = element_text(size=base_size),
                axis_title_y      = element_text(size=base_size, margin={'l': 0, 'r': 0}),
                axis_text_x       = element_text(size=base_size, margin={'t': 12}),
                axis_text_y       = element_text(size=base_size, margin={'r': 12}),
                axis_ticks_length = 0,
                axis_ticks        = element_line(size=0.5),
                # panel setuo
                # panel_background = element_rect(fill=bgcolor),
                # panel_border     = element_line(color='#000000', linetype='solid', size=0.1),
                panel_grid_major = element_blank(),
                panel_grid_minor = element_blank(),
                panel_spacing    = 0.01,

                # plot setup
                # plot_background = element_rect(fill=bgcolor, color=bgcolor, size=1),
                plot_margin=0.025,

                # background
                strip_background=element_rect(fill='#FFFFFF', size=0.2)
                ),
            # inplace=True
            )

def throughput_vs_cores(machine, df, graph='compare-locks'):

    # df['config'] = df['bench_name'] + [str(i) for i in df['n_replicas'].to_list()]
    bench_cat = pd.api.types.CategoricalDtype(categories=CATEGORIES, ordered=True)
    df['bench_name'] = df['bench_name'].astype(bench_cat)
    df = df.loc[df['n_threads'] >= NUMA_NODES]
    df['n_replicas'] = pd.Categorical(df.n_replicas)
    df['write_ratio'] = 100 - df['reads_pct']

    # if graph == 'compare-locks':
    #df = df.loc[~df['bench_name'].isin(['dafny_nr', 'rust_nr'])
    #                    | (df['n_replicas'] == 4)]
    aest = aes(x='n_threads', y='ops_per_s', color='bench_name', shape='bench_name')
    labels = LABELS
    breaks = BREAKS
    linetypes = ['solid', 'dashed', 'dotted']
    # else:
    #     df = df.loc[df['bench_name'].isin(['dafny_nr', 'rust_nr'])]
    #     bench_cat = pd.api.types.CategoricalDtype(categories=['dafny_nr', 'rust_nr'], ordered=True)
    #     df['bench_name'] = df['bench_name'].astype(bench_cat)
    #     n_replicas_cat = pd.api.types.CategoricalDtype(categories=[1, 2, 4], ordered=True)
    #     df['n_replicas'] = df['n_replicas'].astype(n_replicas_cat)
    #     aest = aes(x='n_threads', y='ops_per_s', color='bench_name', shape='bench_name', linetype='n_replicas', group='config')
    #     labels = ['Dafny NR', 'Rust NR']
    #     linetypes = ['dotted', 'dashed', 'solid']
    replicas_labels = ['1 System-wide Replica', '2 Replicas', '4 Replicas (One per NUMA node)']

    print(df)

    def write_ratio_labeller(s):
        return '%d%% writes' % int(s)

    xskip = int(machine[1]/4)
    p = (
        ggplot(data=df, mapping=aest) +
        theme_my538() +
        coord_cartesian(ylim=(0, None), expand=False) +
        labs(y=None) +
        theme(legend_position='top', legend_title=element_blank()) +
        scale_x_continuous(
            breaks=[1, 4] + list(range(xskip, 513, xskip)),
            name='threads') +
        scale_y_continuous(
            name='Mops/sec',
            labels=lambda lst: ["{:,.0f}".format(x / 1_000_000) for x in lst]) +

        scale_shape_manual(values=MARKERS,labels=labels, breaks=breaks, name = "legend") +
        scale_color_manual(values=COLORS, labels=labels, breaks=breaks, name = "legend") +

        #scale_linetype_manual(linetypes, labels=replicas_labels, size=0.2) +
        geom_point(size=0.5) +
        geom_line(size=0.1) +
        #stat_summary(fun_data='mean_sdl', fun_args={'mult': 1}, geom='errorbar') +
        #stat_summary(fun_ymin=np.min, fun_ymax=np.max, geom='errorbar', size=0.1) +
        #stat_summary(fun_y=np.median, geom='line', size=0.1) +
        #stat_summary(fun_y=np.median, geom='point', size=0.05) +
        #stat_summary(fun_y=np.median, geom='point') +
        facet_wrap(["write_ratio"],
                    scales="free",
                    labeller=labeller(cols=write_ratio_labeller))
#guides(color=guide_legend(nrow=1))
    )


    phys_cores = machine[1] / 2
    annotation_data = []
    bench_name = df['bench_name'].unique()[0]
    # config = df['config'].unique()[0]
    for wr in df['write_ratio'].unique():
        annotation_data.append(
            ["", bench_name, wr, phys_cores,
                max(df.loc[df['write_ratio'] == wr]['ops_per_s']), 'A'])

    annotations = pd.DataFrame(annotation_data, columns=[
                                'config', 'bench_name', 'write_ratio', 'n_threads', 'yend', 'lt'])
    annotations['bench_name'] = annotations['bench_name'].astype(pd.api.types.CategoricalDtype())
    annotations['write_ratio'] = annotations['write_ratio'].astype(pd.api.types.CategoricalDtype())
    annotations['config'] = annotations['config'].astype(pd.api.types.CategoricalDtype())
    annotations['lt'] = annotations['lt'].astype(pd.api.types.CategoricalDtype())

    p += geom_segment(data=annotations,
                    mapping=aes(x='n_threads', xend='n_threads',
                                y=0, yend='yend', linetype='lt'),
                    color='black',
                    size=0.1)
    p += scale_linetype_manual(values=['dotted', 'dashed'], guide=None)

    print('saving...')

    p.save("%s-throughput-vs-cores-%s.png" % (machine[0], graph),
           dpi=300, width=PLOT_WIDTH, height=PLOT_HEIGHT,
           units=PLOT_SIZE_UNIT)
    p.save("%s-throughput-vs-cores-%s.pdf" % (machine[0], graph),
           dpi=300, width=PLOT_WIDTH, height=PLOT_HEIGHT,
           units=PLOT_SIZE_UNIT)


def read_data(numa_policy = "fill"):
    df_ironsync = pd.read_json('data-ironsync.json')
    df_ironsync['stdev'] = 0.0 # don't know that one

    if numa_policy == "fill":
        df_ironsync = df_ironsync.loc[df_ironsync['numa_policy'] == 0]
        df_ironsync['numa_policy'] = "NUMAFill"
    else :
        df_ironsync = df_ironsync.loc[df_ironsync['numa_policy'] == 1]
        df_ironsync['numa_policy'] = "Interleave"

    # df_upstream = pd.read_json("data-upstream.json")
    # df_upstream['bench_name'] = 'upstream_nr'
    # if numa_policy == "fill":
    #     df_upstream = df_upstream.loc[df_upstream['numa_policy'] == "NUMAFill"]
    # else:
    #     df_upstream = df_upstream.loc[df_upstream['numa_policy'] == "Interleave"]


    #df_upstream['ops_per_s'] = df_upstream['ops_per_s'] * df_upstream['n_threads']

    df_verus = pd.read_json("data-verified.json")
    df_verus['bench_name'] = 'verus_nr'
    if numa_policy == "fill":
        df_verus = df_verus.loc[df_verus['numa_policy'] == "NUMAFill"]
    else:
        df_verus = df_verus.loc[df_verus['numa_policy'] == "Interleave"]

    #df_verus['ops_per_s'] = df_verus['ops_per_s'] * df_verus['n_threads']

    # df = pd.concat([df_upstream, df_verus, df_ironsync])
    df = pd.concat([df_verus, df_ironsync])
    df = df.loc[df['bench_name'].isin(CATEGORIES)]

    df = df[[
        'bench_name',
        'n_threads',
        'reads_pct',
        'n_replicas',
        'run_seconds',
        'numa_policy',
        'total_ops',
        'ops_per_s',
        'stdev'
    ]]

    df = df.sort_values(by=['numa_policy', 'reads_pct', 'n_threads', 'bench_name'])

    dfs = [ df.loc[df['bench_name'] == c] for c in CATEGORIES]
    df = pd.concat(dfs)

    return df

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.expand_frame_repr', True)

    df = read_data("fill")
    throughput_vs_cores(MACHINE, df.copy(), 'numa-fill')

    print("--------------------------------")

    # df = read_data("interleave")
    # throughput_vs_cores(MACHINE, df.copy(), 'numa-interleave')

    # for machine in MACHINES:
    #     # df = pd.read_json('data-ironsync.json')
    #     throughput_vs_cores(machine, df.copy(), 'compare-locks')
    #     #throughput_vs_cores(machine, df.copy(), 'compare-nrs')
