#rast : imageWriter.o rastByTri.o main.o
#	g++ -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -o rast imageWriter.o rastByTri.o main.o

#imageWriter.o : imageWriter.cpp 
#	g++ -c -o $@ $<

#rastByTri.o : rastByTri.cpp 
#	g++ -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -c -o $@ $<

#main.o : main.cpp 
#	g++ -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -c -o $@ $<

CC = /packages/cuda/11.5.1/bin/nvcc 
source = $(wildcard *.cu)
objects = $(addsuffix .o, $(basename $(source)))
#flags = --x cu -O3 -W -Wall -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DDEBUG=0
flags = -g -G -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DDEBUG=0
target = rast

$(target) : $(objects)
	$(CC) $(flags) -o $(target) $(objects)

%.o : %.cu
	$(CC) $(flags) -c $< -o $@

clean :
	rm $(target) $(wildcard *.o)
