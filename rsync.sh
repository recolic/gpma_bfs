#!/bin/bash

remote=49.232.73.57

[[ $1 = pull ]] && rsync -avz -zz --progress   root@$remote:/root/gpma_bfs/ ./
[[ $1 = push ]] && rsync -avz -zz --progress ./ root@$remote:/root/gpma_bfs/

