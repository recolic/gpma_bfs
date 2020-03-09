#!/bin/bash

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
        edges.append((a, b))
        node_size = max(node_size, a)
        node_size = max(node_size, b)
edge_size = len(edges)
random.shuffle(edges)
with open('pokec.txt', 'w') as f:
    f.write('{} {}\n'.format(node_size, edge_size))
    for a, b in edges:
        f.write('{} {}\n'.format(a - 1, b - 1))
print('pokec.txt done.')
" > /tmp/gpma-tmp.py

python /tmp/gpma-tmp.py
exit $?

