#! /usr/bin/bash

read -p "File: " fin

base=$(basename $fin .stl)
fout="img/${base}.pnm"
echo $fin
echo $fout

#./triCount $fin

echo "Scale, Subdivisions, Rasterize, Runtime"

for i in 1 2 4 8 16 32 64 128
do
	for j in 0 1 2 3 4 5 6
	do
		echo -n "$i, $j"
		build_viskores/rast --viskores-device=Kokkos $fin $fout $i $j
	done
done
