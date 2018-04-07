all: 
	 nvcc -arch=sm_35 -rdc=true -O3 -std=c++11 -Xcompiler -fopenmp -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI *.cu -o exec

Main.o: Main.cu
	g++ -c -fopenmp Main.cu

CSR.o: CSR.cpp
	g++ -c -fopenmp CSR.cpp 

Ellpack.o: Ellpack.cpp
	g++ -c -fopenmp Ellpack.cpp
clean:
	rm -rf *.o 
	rm exec 
