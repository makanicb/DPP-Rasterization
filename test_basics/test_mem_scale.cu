#include <iostream>
#include <functional>
#include <cmath>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/Timer.h>
#include <viskores/worklet/WorkletMapField.h>

struct ExpensiveWorklet : viskores::worklet::WorkletMapField
{
	using ControlSignature = void(FieldIn a, FieldIn b, FieldOut c);
	using ExecutionSignature = _3(_1, _2);
	using InputDomain = _1;

	VISKORES_EXEC int operator()(const int a, const int b) const
	{
		int prod = 1;
		int mod = 4747;
		for(int i = 0; i < a * 1e4; i++)
		{
			prod = (prod * b) % mod;
		}

		return prod;
	}
};

int main(int argc, char **argv){

	//Initialize program
	viskores::cont::Initialize(argc, argv, viskores::cont::InitializeOptions::AddHelp);
	bool first = true;
	for(int j = 0; j <= 20; j+=2)
	{
		const int ARRAY_SIZE = (int) pow(2, j);

		int start = first ? 0 : 1;
		first = false;
		for(int i = start; i < 2; i++)
		{
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

			if(i == 1)
			{
				//Print runtime
				std::cout << ARRAY_SIZE <<
					", " << timer.GetElapsedTime() * 1e6 << std::endl;
			}
		}
	}
	return 0;
}
