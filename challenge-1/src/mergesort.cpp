/*
*  This file is part of Christian's OpenMP software lab
*
*  Copyright (C) 2016 by Christian Terboven <terboven@itc.rwth-aachen.de>
*  Copyright (C) 2016 by Jonas Hahnfeld <hahnfeld@itc.rwth-aachen.de>
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <cstring>
#include <omp.h>

using std::log2, std::upper_bound;


/**
  * helper routine: check if array is sorted correctly
  */
bool isSorted(int ref[], int data[], const size_t size){
	std::sort(ref, ref + size);
	for (size_t idx = 0; idx < size; ++idx){
		if (ref[idx] != data[idx]) {
			return false;
		}
	}
	return true;
}


/**
  * sequential merge step (straight-forward implementation)
  */
void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin) {
	// there are some optimizations here, but they are just syntactic sugar
	long left = begin1, right = begin2, idx = outBegin;

	while (left < end1 && right < end2) {
		out[idx++] = in[left] <= in[right] ? in[left++] : in[right++];
	}

	while (left < end1) {
		out[idx++] = in[left++];
	}

	while (right < end2) {
		out[idx++] = in[right++];
	}
}

/**
  * sequential merge step (straight-forward implementation);
  * it is a parallel implementation; parallelization depends on depth (like MsParallel function);
  * depth > 0, parallelize, otherwise use the sequential version.
  */
void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin, int depth) {
	// depth greater than zero, then use threads
	if (depth > 0) {
		// take the mid-index of the first array (1)
		const long mid1 = (begin1 + end1) / 2;
		// take the second mid-index of the second array;
		// to optimize this, we use some kind of binary search (not at all);
		// the upper_bound function finds the first element in a sorted array that is greater than a given value;
		// we have the guarantee that the array is ordered because this function is called after sorting;
		// so take the first position using the offset memory of the in pointer,
		// take the last position using the offset memory again;
		// the value to search for is the actual mid-value of the array.
		// It works like a sorting algorithm;
		// it takes a value from the array and
		// with the search tries to find the next value that is just greater than the given one.
		const long mid2 = upper_bound(in + begin2, in + end2, in[mid1]) - in;
		// merge the two parts in parallel
		--depth;
		#pragma omp task
		MsMergeSequential(out, in, begin1, mid1, begin2, mid2, outBegin, depth);

		#pragma omp task
		// is the outBegin index used by the merge function:
		// - the outBegin index;
		// - plus, the start minus the end of the first part of the previous merge
		// - plus, the start minus the end of the second part of the previous merge
		MsMergeSequential(out, in, mid1, end1, mid2, end2, outBegin + (mid1 - begin1) + (mid2 - begin2), depth);

		// synchronization
		#pragma omp taskwait
	} else {
		// sequential version
		MsMergeSequential(out, in, begin1, end1, begin2, end2, outBegin);
	}
}


/**
  * sequential MergeSort
  */
void MsSequential(int *array, int *tmp, bool inplace, long begin, long end) {
	// original sequential version, is the same
	if (begin < end - 1) {
		const long half = (begin + end) / 2;
		MsSequential(array, tmp, !inplace, begin, half);
		MsSequential(array, tmp, !inplace, half, end);
		if (inplace) {
			MsMergeSequential(array, tmp, begin, half, half, end, begin);
		} else {
			MsMergeSequential(tmp, array, begin, half, half, end, begin);
		}
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
}


/**
  * parallel OpenMP MergeSort
  */
void MsParallel(int *array, int *tmp, bool inplace, long begin, long end, int depth, int maxDepth) {
	if (begin < end - 1) {
		// take the half index
		const long half = (begin + end) / 2;

		// if the depth is greater than zero, no overhead, then we can use OpenMP to parallelize
		if (depth > 0) {
			// the call is recursive, so as we go deeper in the tree, the depth index is updated
			// first part of the array, from the begin to the half;
			--depth;
			#pragma omp task
			MsParallel(array, tmp, !inplace, begin, half, depth, maxDepth);
			// seconod part of the array, from the half to the end
			#pragma omp task
			MsParallel(array, tmp, !inplace, half, end, depth, maxDepth);
			// synchronization
			#pragma omp taskwait
		} else {
			// maximum depth, then we use the sequential mode
			MsSequential(array, tmp, !inplace, begin, half);
			MsSequential(array, tmp, !inplace, half, end);
		}
		// if we decide to use inplace way
		if (inplace) {
			MsMergeSequential(array, tmp, begin, half, half, end, begin, maxDepth);
		} else {
			MsMergeSequential(tmp, array, begin, half, half, end, begin, maxDepth);
		}
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
}


/**
  * Serial MergeSort
  */
void MsSerial(int *array, int *tmp, const size_t size) {
	// set the limit to avoid too few tasks
	constexpr int cut_off = 10000;
	// set the maximum depth;
	// it is equal to the base 2 logarithm of the number of threads available in the machine;
	// the base 2 logarithm is chosen because the merge sort "splits"
	// each part into two shelves at each iteration
	const int maxDepth = static_cast<int>(log2(omp_get_max_threads()));
	// set the maximum depth considering the size of the array;
	// it is the size of the array divided by the size of the cut_off,
	// the result is the number of partitions we can create in the original array with a given cut_off;
	// now the base 2 logarithm is made to guarantee the maximum depth of recursion
	// (as we did with the max depth, using the same logic)
	const int maxDepthBySize = static_cast<int>(log2(size / cut_off));

	// since we have two limits, which one should we choose?
	// - maxDepth identifies the recursion limit based on the number of threads available in the machine;
	// - maxDepthBySize identifies the recursion limit based on the input array size and the cut-off set.
	// the answer is: the minimum;
	// if the depth given by the size/cut-off is too high, we use all our threads,
	// otherwise only the minimum without resource overhead;
	// note that if the size is equal to or less than the cut-off,
	// the minimum value taken is zero (log operation returns a negative or zero value);
	// this means that in the parallel function, the thread #0 will always execute the sequential algorithm
	int optimalDepth = std::min(maxDepth, maxDepthBySize);
	if (optimalDepth < 0) {
		optimalDepth = 0;
	}

	// create a parallel region
	#pragma omp parallel
	#pragma omp single // forces only one thread
	MsParallel(
		array, // res array
		tmp, // array used to inplace
		true, // use inplace
		0, // initially consider the entire array, from zero to the end
		size,
		optimalDepth, // initially use the optimal depth calculated
		optimalDepth // set the maximum depth to the logarithmic result
	);
}


/**
  * @brief program entry point
  */
int main(int argc, char* argv[]) {
	// variables to measure the elapsed time
	struct timeval t1, t2;
	double etime;

	// expect one command line arguments: array size
	if (argc != 2) {
		printf("Usage: MergeSort.exe <array size> \n");
		printf("\n");
		return EXIT_FAILURE;
	}
	else {
		const size_t stSize = strtol(argv[1], NULL, 10);
		int *data = (int*) malloc(stSize * sizeof(int));
		int *tmp = (int*) malloc(stSize * sizeof(int));
		int *ref = (int*) malloc(stSize * sizeof(int));

		printf("Initialization...\n");

		srand(95);
		for (size_t idx = 0; idx < stSize; ++idx){
			data[idx] = (int) (stSize * (double(rand()) / RAND_MAX));
		}
		std::copy(data, data + stSize, ref);

		double dSize = (stSize * sizeof(int)) / 1024 / 1024;
		printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

		gettimeofday(&t1, NULL);
		MsSerial(data, tmp, stSize);
		gettimeofday(&t2, NULL);

		etime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
		etime = etime / 1000;

		printf("done, took %f sec. Verification...", etime);
		if (isSorted(ref, data, stSize)) {
			printf(" successful.\n");
		}
		else {
			printf(" FAILED.\n");
		}

		free(data);
		free(tmp);
		free(ref);
	}

	return EXIT_SUCCESS;
}