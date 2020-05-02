#!/usr/bin/fish

for i in (seq 8)
    echo CPUS=$i ===========================================
    make EXTRA_FLAGS="-DTEST_CPUS=$i -DTEST_GPUS=0 -DTEST_DISABLE_BFS"
    and time ./gpma_bfs_demo /tmp/999999999.pokec.txt 0
end



