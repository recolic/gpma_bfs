#!/bin/bash

remote=49.232.74.97

[[ $1 = pull ]] && rsync -avz -zz --progress   root@$remote:/root/gpma_bfs/ ./
[[ $1 = push ]] && rsync -avz -zz --progress ./ root@$remote:/root/gpma_bfs/

