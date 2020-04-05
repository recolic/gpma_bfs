#!/usr/bin/fish
function t
    make TEST_DEV=$argv[1]
    and ./gpma_bfs_demo /dataset/10000.pokec.txt 0 | tee /dev/fd/2 2>| grep 8261

    set ret $status
    test $ret = 0 ; and echo $argv[1] OK ; or echo $argv[1] FAILED
    return $ret
end

t CPU
and t GPU

exit $status

