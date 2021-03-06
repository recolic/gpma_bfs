#!/bin/bash
# Usage: ./prepare.sh 100
# Usage: ./prepare.sh 10000
# Usage: ./prepare.sh 999999999999

NODE_MAX="$1"
[[ $NODE_MAX = "" ]] && NODE_MAX=999999999

# download dataset
if [[ ! -f soc-pokec-relationships.txt ]]; then
	echo 'downloading graph dataset...'
	wget https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz &&
	gzip -d soc-pokec-relationships.txt.gz || exit $?
fi

echo "
import os
import random
# reformat and shuffle
print('re-formating and shuffling graph dataset...')
node_size = 0
edges = []
with open('soc-pokec-relationships.txt', 'r') as f:
    for line in f.readlines():
        a, b = line.strip().split('\t')
        a, b = int(a), int(b)
        if a > $NODE_MAX or b > $NODE_MAX:
            continue
        edges.append((a, b))
        node_size = max(node_size, a)
        node_size = max(node_size, b)
edge_size = len(edges)
random.shuffle(edges)
with open('$NODE_MAX.pokec.txt', 'w') as f:
    f.write('{} {}\n'.format(node_size, edge_size))
    for a, b in edges:
        f.write('{} {}\n'.format(a - 1, b - 1))
print('pokec.txt done.')
" > /tmp/gpma-tmp.py

python /tmp/gpma-tmp.py
exit $?

