#! /usr/bin/bash

read -p "File: " fin

base=$(basename $fin .stl)
fout="img/${base}.pnm"
echo $fin
echo $fout

./triCount $fin

echo -e "Scale\tSubdivisions\tRasterize\tSort\tSelect\tWrite"

for i in 1 2 4 8 16
do
	for j in 0 1 2 3 4
	do
		echo -n -e "$i\t$j"
		build/rast --viskores-device CUDA $fin $fout $i $j
	done
done
