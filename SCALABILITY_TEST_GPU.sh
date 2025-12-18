#! /usr/bin/bash

read -p "File: " fin

base=$(basename $fin .stl)
fout="img/${base}.pnm"
echo $fin
echo $fout

echo -e "Scale, Rasterize, Sort, Select, Write"

for i in 1 2 4 8 16
do
	echo -n -e "$i\t$mult"
	build_thrust/rast $fin $fout $i
done

echo -e "Subdivisions, Rasterize, Sort, Select, Write"

for i in 0 1 2 3 4 5 6
do
	echo -n -e "$i"
	build_thrust/rast $fin $fout 1 $i
done
