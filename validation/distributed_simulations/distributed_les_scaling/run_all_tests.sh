#!/bin/bash

for rX in 1 2 4 8 16 32 64; do
    for rY in 1 2 4 8 16 32 64; do
        export RX=$rX
        export RY=$rY

        echo "Running distributed LES on $((RX * $RY)) processors" 
        ./run_les.sh
    done
done
