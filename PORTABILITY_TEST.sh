#! /usr/bin/bash

read -p "Models: " models
read -p "Images: " img

export OMP_NUM_THREADS=14

for fin in "$models"/*.stl; do
	base=$(basename $fin .stl)
	fout="$img/${base}.pnm"
	echo -n $base
	./rast $fin $fout 10
done


#base=$(basename $fin .stl)
#fout="${base}.pnm"
#echo $fin
#echo $fout

#./triCount $fin

#for i in 1 2 4 8 14
#do
#	echo -n $i
#	export OMP_NUM_THREADS=$i
#	./rast $fin $fout
#done
