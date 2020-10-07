#!/bin/sh

for i in {1..49} 
do 
python src/RiWalk-RW/RiWalk-RW.py --input graphs/elliptic_t$i.edgelist --output embs/elliptic_t$i.emb --dimensions 128 --num-walks 10 --walk-length 80 --window-size 10 --workers 8 --iter 10 --flag wl
done  
