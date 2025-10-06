#include<iostream>
#include<cmath>

#include<thrust/device_vector.h>
#include<thrust/fill.h>
#include<thrust/functional.h>
#include<thrust/transform.h>

struct expensiveFunctor
{
	__host__ __device__
	int operator()(const int a, const int b) const
	{
		int prod = 1;
		int mod = 4747;
		for(int i = 0; i < a * 1e3; i++)
		{
			prod = (prod * b) % mod;
		}
		return prod;
	}
};

int main(int argc, char **argv)
{
	bool first = true;
	for(int j = 0; j <= 20; j+=4)
	{
		//Initialize program
		const unsigned int ARRAY_SIZE = pow(2, j);

		int start = first ? 0 : 1;
		first = false;
		for(int i = start; i < 2; i++)
		{
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
			thrust::transform(arr2.begin(), arr2.end(), arr1.begin(), sum.begin(), expensiveFunctor());

			//Stop timer
			cudaEventRecord(stop);

			if(i == 1)
			{
				//Print runtime
				cudaEventSynchronize(stop);
				float runtime_ms = 0;
				cudaEventElapsedTime(&runtime_ms, start, stop);
				std::cout << ARRAY_SIZE
					<< ", " << runtime_ms * 1e3 << std::endl;
			}
		}
	}

	return 0;
}
