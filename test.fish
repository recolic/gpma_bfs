#!/usr/bin/fish
function t
    make TEST_DEV=$argv[1]
    and ./gpma_bfs_demo /dataset/999999999.pokec.txt 0 | tee /dev/fd/2 2>| grep 1334630

    set ret $status
    test $ret = 0 ; and echo $argv[1] OK ; or echo $argv[1] FAILED
    return $ret
end

t CPU
and t GPU

exit $status

