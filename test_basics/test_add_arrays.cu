#include<iostream>
#include<cmath>

#include<thrust/device_vector.h>
#include<thrust/fill.h>
#include<thrust/functional.h>
#include<thrust/transform.h>

#ifndef EXP
#define EXP 8
#endif

int main(int argc, char **argv)
{
	//Initialize program
	const int ARRAY_SIZE = pow(2, EXP);

	//Initialize timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Start timer
	cudaEventRecord(start);

	//Initialize arrays
	thrust::device_vector<int> arr1(ARRAY_SIZE);
	thrust::device_vector<int> arr2(ARRAY_SIZE);
	thrust::fill(arr1.begin(), arr1.end(), 256);
	thrust::fill(arr2.begin(), arr2.end(), 1024);

	//Add arrays
	thrust::device_vector<int> sum(ARRAY_SIZE);
	thrust::transform(arr1.begin(), arr1.end(), arr2.begin(), sum.begin(), thrust::plus<int>());

	//Stop timer
	cudaEventRecord(stop);

	//Print runtime
	cudaEventSynchronize(stop);
	float runtime_ms = 0;
	cudaEventElapsedTime(&runtime_ms, start, stop);
	std::cout << "Time (microseconds) to add two arrays (len=" << ARRAY_SIZE
		<< "): " << runtime_ms * 1e3 << std::endl;

	return 0;
}
