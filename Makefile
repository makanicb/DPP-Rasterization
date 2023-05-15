#rast : imageWriter.o rastByTri.o main.o
#	g++ -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -o rast imageWriter.o rastByTri.o main.o

#imageWriter.o : imageWriter.cpp 
#	g++ -c -o $@ $<

#rastByTri.o : rastByTri.cpp 
#	g++ -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -c -o $@ $<

#main.o : main.cpp 
#	g++ -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -c -o $@ $<

source = $(wildcard *.cpp)
objects = $(addsuffix .o, $(basename $(source)))
flags = -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
target = rast

$(target) : $(objects)
	g++ $(flags) -o $(target) $(objects)

%.o : %.cpp
	g++ $(flags) -c $< -o $@

clean :
	rm $(target) $(wildcard *.o)
