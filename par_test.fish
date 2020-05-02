#!/usr/bin/fish

set g 1
for f in (seq 0 20)
    echo FACTOR=$f ===========================================
    make EXTRA_FLAGS="-DTEST_GPU_FACTOR=$f -DTEST_DISABLE_BFS"
    and time ./gpma_bfs_demo /tmp/100000.pokec.txt 0
end

exit 0

set g 0
for i in (seq 8)
    echo CPUS=$i GPUS=$g ===========================================
    make EXTRA_FLAGS="-DTEST_CPUS=$i -DTEST_GPUS=$g -DTEST_DISABLE_BFS"
    and time ./gpma_bfs_demo /tmp/999999999.pokec.txt 0
end



