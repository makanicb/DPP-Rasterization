#include <iostream>
#include <functional>
#include <cmath>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/Timer.h>
#include <viskores/worklet/WorkletMapField.h>

#ifndef EXP
#define EXP 8
#endif

struct ExpensiveWorklet : viskores::worklet::WorkletMapField
{
	using ControlSignature = void(FieldIn a, FieldIn b, FieldOut c);
	using ExecutionSignature = _3(_1, _2);
	using InputDomain = _1;

	VISKORES_EXEC int operator()(const int a, const int b) const
	{
		int prod = 1;
		int mod = 4747;
		for(int i = 0; i < a * 1e6; i++)
		{
			prod = (prod * b) % mod;
		}

		return prod;
	}
};

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

	//Apply the expensive worklet to the array
	viskores::cont::ArrayHandle<int> sum;
	viskores::cont::Invoker invoke;
	ExpensiveWorklet expensive_worklet;
	invoke(expensive_worklet, arr2, arr1, sum);

	//Stop timer
	timer.Stop();

	//Print runtime
	std::cout << "Time (microseconds) to add arrays (len=" << ARRAY_SIZE <<
		"): " << timer.GetElapsedTime() * 1e6 << std::endl;

	return 0;
}
