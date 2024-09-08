#!/usr/bin/env python3

import sys
import csv
import json
import os
import os.path
import glob
import subprocess
import random
import time
import math
import re

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

SECONDS = 60

MODES=[
    'fill',
    'interleave'
]
CORES_PER_NODE = count_cores_per_numa_node()
NODES = count_numa_nodes()
MAX_THREADS = NODES * CORES_PER_NODE
N_THREADS = [MAX_THREADS, 4] + [i * (CORES_PER_NODE // 4) for i in range(1, NODES * 4)]
print(f"Used Threads: {N_THREADS}")

NR_BENCHES = ['cargo run --release --bin vspace']
#READS_PCT = [100, 95, 50, 0, 90]
READS_PCT = [100, 90, 0]

ITERS = 1

TRANSPARENT_HUGEPAGES = True


def combine_data_files():
    filepaths = glob.glob('nr_benchmarks*.json')
    json_records = []
    for filepath in filepaths:
        with open(filepath) as f:
            json_records.append(f.read())

    combined_json = '[\n' + ',\n'.join(json_records) + '\n]'

    with open('data.json', 'w') as f:
        f.write(combined_json)

def run(bench, n_replicas, n_threads, reads_pct, run_id_num, numa_policy):
    cmd = '%s %d %d %d %s %s' % (bench, n_threads, reads_pct,
                                   SECONDS, numa_policy, run_id_num)
    print(cmd)
    # squash warnings...
    my_env = os.environ
    my_env["RUSTFLAGS"] = "-Awarnings"
    subprocess.run(cmd, shell=True, check=False, env=my_env)

def run_all():
    print(f'Found {NODES} NUMA nodes with {CORES_PER_NODE} cores each')

    subprocess.run('sudo sh -c "echo %s > /sys/kernel/mm/transparent_hugepage/enabled"' % ('never', 'always')[TRANSPARENT_HUGEPAGES], shell=True, check=False)
    for nid in range(0, NODES):
        subprocess.run('sudo sh -c "echo 16 > /sys/devices/system/node/node{}/hugepages/hugepages-1048576kB/nr_hugepages"'.format(nid), shell=True, check=False)

    subprocess.run('rm -rf data*.json nr_benchmarks*.json nr_benchmarks*.csv *throughput*.pdf *throughput*.png', shell=True, check=False)

    try:
        os.mkdir('runs')
    except:
        pass

    run_id = time.strftime('run%Y%m%d%H%M%S')
    try:
        os.mkdir(os.path.join('runs', run_id))
    except:
        pass

    run_num = -1
    for i in range(ITERS):
        run_num += 1
        run_id_num = run_id + '-' + str(run_num)
        for reads_pct in READS_PCT:
            for n_threads in N_THREADS:
                n_replicas = min(math.ceil(n_threads / (CORES_PER_NODE / 2)), NODES)
                for mode in MODES:
                    for bench in NR_BENCHES:
                        if (n_threads < n_replicas):
                            continue
                        run(bench, n_replicas, n_threads, reads_pct, run_id_num, mode)

                    combine_data_files()
                    subprocess.run('cp *.json runs/%s' % run_id, shell=True, check=False)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--node-count':
            print(NODES)
        else:
            print('Unknown option')
    else:
        run_all()
