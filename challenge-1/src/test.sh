#!/bin/bash

i=$1

g++ -O3 -fopenmp mergesort.cpp -o merge -fopenmp
g++ -O3 -fopenmp sequential_mergesort.cpp -o seq_merge -fopenmp

echo "SEQUENTIAL: "
./seq_merge $i
echo "PARALLEL: "
./merge $i
echo ""

