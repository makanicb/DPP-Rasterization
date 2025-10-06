#include <iostream>
#include <functional>
#include <cmath>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/Timer.h>

int main(int argc, char **argv){

	//Initialize program
	viskores::cont::Initialize(argc, argv, viskores::cont::InitializeOptions::AddHelp);
	int start = 0;
	for(int i = 0; i <= 30; i++)
	{
		const int ARRAY_SIZE = (int) pow(2, i);

		for(int j = start; j < 2; j++)
		{
			//Initialize and start timer
			viskores::cont::Timer timer;	
			timer.Start();

			for(int k = 0; k < 100; k++)
			{
				//Initialize arrays
				viskores::cont::ArrayHandle<int> arr1;
				viskores::cont::ArrayHandle<int> arr2;
				arr1.AllocateAndFill(ARRAY_SIZE, 256);
				arr2.AllocateAndFill(ARRAY_SIZE, 1024);

				//Take the sum of the arrays
				viskores::cont::ArrayHandle<int> sum;
				viskores::cont::Algorithm::Transform(arr1, arr2, sum, std::plus<int>());
			}

			//Stop timer
			timer.Stop();

			if(j == 1)
			{
				//Print runtime
				std::cout << ARRAY_SIZE <<
					", " << timer.GetElapsedTime() * 1e6 << std::endl;
			}
		}
		start = 1;
	}

	return 0;
}
