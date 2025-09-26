#include <iostream>
#include <functional>

#include <viskores/cont/ArrayHandle>

int main(int argc, char **argv){

	const int ARRAY_SIZE = 1 << 32;

	//Initialize arrays
	viskores::cont::ArrayHandle<int> arr1;
	viskores::cont::ArrayHandle<int> arr2;
	arr1.AllocateAndFill(ARRAY_SIZE, 256);
	arr2.AllocateAndFill(ARRAY_SIZE, 1024);

	//Take the sum of the arrays
	viskores::cont::ArrayHandle<int> sum;


	return 0;
}
