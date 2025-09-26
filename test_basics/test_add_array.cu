#include <iostream>
#include <functional>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/Initialize.h>

int main(int argc, char **argv){

	viskores::cont::Initialize(argc, argv, viskores::cont::InitializeOptions::AddHelp);
	const int ARRAY_SIZE = 1 << 32;

	//Initialize arrays
	viskores::cont::ArrayHandle<int> arr1;
	viskores::cont::ArrayHandle<int> arr2;
	arr1.AllocateAndFill(ARRAY_SIZE, 256);
	arr2.AllocateAndFill(ARRAY_SIZE, 1024);

	//Take the sum of the arrays
	viskores::cont::ArrayHandle<int> sum;
	viskores::cont::Algorithm::Transform(arr1, arr2, sum, std::plus<int>());

	return 0;
}
