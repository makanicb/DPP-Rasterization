#! /usr/bin/bash

read -p "File: " fin

base=$(basename $fin .tri)
fout="${base}.pnm"
echo $fin
echo $fout

for i in 1 2 4 8 14
do
	echo -n $i
	export OMP_NUM_THREADS=$i
	./rast $fin $fout
done
