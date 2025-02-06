#! /usr/bin/bash

read -p "File: " fin

base=$(basename $fin .stl)
fout="${base}.pnm"
echo $fin
echo $fout

./triCount $fin

for i in 1 2 4 8 14
do
	echo -n $i
	export OMP_NUM_THREADS=$i
	./rast $fin $fout
done
