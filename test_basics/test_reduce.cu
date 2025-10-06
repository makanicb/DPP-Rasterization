#include<iostream>
#include<cmath>
#include<cstdlib>

#include<thrust/device_vector.h>
#include<thrust/fill.h>
#include<thrust/functional.h>
#include<thrust/transform.h>

int main(int argc, char **argv)
{
	//Initialize program
	int exp = 8;
	if(argc >= 2) exp = atoi(argv[1]);
	const unsigned int ARRAY_SIZE = (unsigned int) pow(2, exp);

	for(int i = 0; i < 2; i++)
	{
		//Initialize timer
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//Start timer
		cudaEventRecord(start);

		//Initialize arrays
		thrust::device_vector<double> arr(ARRAY_SIZE);
		thrust::fill(arr.begin(), arr.end(), 1024.0);

		//Add arrays
		double sum = thrust::reduce(arr.begin(), arr.end());

		//Stop timer
		cudaEventRecord(stop);

		if(i == 1)
		{
			//Print runtime
			cudaEventSynchronize(stop);
			float runtime_ms = 0;
			cudaEventElapsedTime(&runtime_ms, start, stop);
			std::cout << "Time (microseconds) sum one array (len=" << ARRAY_SIZE
				<< "): " << runtime_ms * 1e3 << std::endl;
		}
	}

	return 0;
}
