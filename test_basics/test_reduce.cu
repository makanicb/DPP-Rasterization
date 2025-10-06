#include <iostream>
#include <functional>
#include <cmath>
#include <cstdlib>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/Timer.h>

int main(int argc, char **argv){

	//Initialize program
	viskores::cont::Initialize(argc, argv, viskores::cont::InitializeOptions::AddHelp);

	int exp = 8;
	if(argc >= 2) exp = atoi(argv[1]);
	const unsigned int ARRAY_SIZE = (unsigned int) pow(2, exp);

	for(int i = 0; i < 2; i++)
	{
		//Initialize and start timer
		viskores::cont::Timer timer;	
		timer.Start();

		//Initialize arrays
		viskores::cont::ArrayHandle<double> arr;
		arr.AllocateAndFill(ARRAY_SIZE, 1024.0);

		//Take the sum of the arrays
		double sum = viskores::cont::Algorithm::Reduce(arr, 0.0);

		//Stop timer
		timer.Stop();

		if(i == 1)
		{
			//Print runtime
			std::cout << "Time (microseconds) to add arrays (len=" << ARRAY_SIZE <<
				"): " << timer.GetElapsedTime() * 1e6 << std::endl;
		}
	}

	return 0;
}
