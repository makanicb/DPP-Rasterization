#! /usr/bin/bash

read -p "File: " fin

base=$(basename $fin .stl)
fout="img/${base}.pnm"
echo $fin
echo $fout

./triCount $fin

for i in 1 2 4 8 14
do
	echo -n $i
	build/rast --viskores-device TBB --viskores-num-threads $i $fin $fout
done
