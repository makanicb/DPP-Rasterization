#include <iostream>
#include <functional>
#include <cmath>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/Timer.h>

#ifndef EXP
#define EXP 8
#endif

int main(int argc, char **argv){

	//Initialize program
	viskores::cont::Initialize(argc, argv, viskores::cont::InitializeOptions::AddHelp);
	const int ARRAY_SIZE = (int) pow(2, EXP);

	//Initialize and start timer
	viskores::cont::Timer timer;	
	timer.Start();

	//Initialize arrays
	viskores::cont::ArrayHandle<int> arr1;
	viskores::cont::ArrayHandle<int> arr2;
	arr1.AllocateAndFill(ARRAY_SIZE, 256);
	arr2.AllocateAndFill(ARRAY_SIZE, 1024);

	//Take the sum of the arrays
	viskores::cont::ArrayHandle<int> sum;
	viskores::cont::Algorithm::Transform(arr1, arr2, sum, std::plus<int>());

	//Stop timer
	timer.Stop();

	//Print runtime
	std::cout << "Time (microseconds) to add arrays (len=" << ARRAY_SIZE <<
		"): " << timer.GetElapsedTime() * 1e6 << std::endl;

	return 0;
}
