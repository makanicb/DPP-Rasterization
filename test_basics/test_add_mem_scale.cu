#include<iostream>
#include<cmath>

#include<thrust/device_vector.h>
#include<thrust/fill.h>
#include<thrust/functional.h>
#include<thrust/transform.h>

int main(int argc, char **argv)
{
	bool first = true;
	for(int i = 0; i <= 30; i++)
	{
		//Initialize program
		const int ARRAY_SIZE = pow(2, i);

		int start = first ? 0 : 1;
		first = false;
		for(int j = start; j < 2; j++)
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
			for(int k = 0; k < 100; k++)
				thrust::transform(arr1.begin(), arr1.end(), arr2.begin(), sum.begin(), thrust::plus<int>());

			//Stop timer
			cudaEventRecord(stop);

			if(j == 1)
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
