#! /usr/bin/bash

read -p "Models: " models
read -p "Images: " img
read -p "Device: " dev

export OMP_NUM_THREADS=24

for fin in "$models"/*.stl; do
        base=$(basename $fin .stl)
        fout="$img/${base}.pnm"
        echo -n $base
        build/rast --viskores-device=$dev $fin $fout 10
done
