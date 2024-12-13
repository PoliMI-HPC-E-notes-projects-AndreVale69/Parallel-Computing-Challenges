# Challenges of Parallel Computing - OpenMP and CUDA

[![OpenMP](https://img.shields.io/badge/OpenMP-5.2-blue.svg)](https://www.openmp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-zone)

## Description

The repository contains a set of challenges to be solved using parallel computing techniques.
The challenges are designed to be solved using the [OpenMP][OpenMP] and [CUDA][CUDA] frameworks.

The challenges are part of the [Parallel Computing][PC_Course] course at the [Politecnico di Milano][POLIMI].

The challenges fall into two main categories:
- [Challenge 1](challenge-1) - OpenMP. This challenge is designed to parallelize 
  the [Merge Sort][MergeSort] algorithm using the OpenMP framework.
- [Challenge 2](challenge-2) - CUDA. The goal of this challenge is to parallelize the 2D convolution algorithm
  using the CUDA framework. Two solutions are proposed: a naive solution using the GPU
  and an advanced solution using shared memory ([tiling technique][tiling]).

------------------------------------------------------------------------------------------------------------------------

## How to run the code

Of course, you need to have the OpenMP and CUDA frameworks installed on your machine to run the following code.
You will also need a compatible GPU to run the CUDA code (otherwise you can use the [Google Colab][GoogleColab]).

If you have [CLion][CLion] installed, this is a simple story. Just open the project and run the code using the provided
[CMakeLists.txt](CMakeLists.txt) file.

Otherwise, you can compile the code using the command line.
1. Compile the CMakeFiles:
   ```bash
   cd Parallel-Computing-Challenges # repository folder
   cmake . # where the CMakeLists.txt file is located
   ```
2. Compile all the challenges using the following commands:
   ```bash
   # assuming you are in the repository folder where the CMakeLists.txt file is located
   make -f ./Makefile -C . all
   ```
3. And finally, run the compiled code:
   ```bash
   # assuming you are in the repository folder where the CMakeLists.txt file is located
   ./openmp-parallel-mergesort 1000 # where 1000 is the size of the array to be sorted
   export MASK_SIZE=3 && export MATRIX_HEIGHT=100 && export MATRIX_WIDTH=100 && ./cuda-2d-convolution
   # and so on...
   ```
4. Clean the compiled files:
   ```bash
   # assuming you are in the repository folder where the CMakeLists.txt file is located
   make -f ./Makefile -C . clean
   ```

You can also compile just one of the challenges using the following commands:
1. Compile the CMakeFiles:
   ```bash
   cd Parallel-Computing-Challenges
   cmake .
   ```
2. Compile only one of the challenges using the following commands:
   ```bash
   # assuming you are in the repository folder where the CMakeLists.txt file is located
   make -f ./Makefile -C . openmp-parallel-mergesort
   ```
   or
   ```bash
   # assuming you are in the repository folder where the CMakeLists.txt file is located
   make -f ./Makefile -C . cuda-2d-convolution
   ```
3. And finally, run the compiled code:
   ```bash
   ./openmp-parallel-mergesort 1000
   ```
   or
   ```bash
   export MASK_SIZE=3 && export MATRIX_HEIGHT=100 && export MATRIX_WIDTH=100 && ./cuda-2d-convolution
   ```

The names of the executables are available in the [CMakeLists.txt](CMakeLists.txt) file.
However, here is a list of available programs:
- [sequential-mergesort](challenge-1/src/sequential_mergesort.cpp)
- [openmp-parallel-mergesort](challenge-1/src/mergesort.cpp)
- [cuda-matrix-multiplication](challenge-2/src/matrix-multiplication.cu)
- [cuda-2d-convolution](challenge-2/src/2d-convolution-basic.cu)
- [cuda-2d-convolution-tiling](challenge-2/src/2d-convolution-tiling.cu)


[OpenMP]: https://www.openmp.org/
[CUDA]: https://developer.nvidia.com/cuda-zone
[PC_Course]: https://github.com/PoliMI-HPC-E-notes-projects-AndreVale69/HPC-E-PoliMI-university-notes/blob/main/parallel-computing
[POLIMI]: https://www.polimi.it/
[MergeSort]: https://en.wikipedia.org/wiki/Merge_sort
[tiling]: https://arxiv.org/pdf/1001.1718
[CLion]: https://www.jetbrains.com/clion/
[GoogleColab]: https://colab.research.google.com/