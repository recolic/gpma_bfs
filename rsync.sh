#!/bin/bash

remote=152.136.220.26

[[ $1 = pull ]] && rsync -avz -zz --progress   root@$remote:/root/gpma_bfs/ ./
[[ $1 = push ]] && rsync -avz -zz --progress ./ root@$remote:/root/gpma_bfs/

