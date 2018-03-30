all: Main.o CSR.o Ellpack.o
	g++ Main.o -fopenmp CSR.o Ellpack.o -o exec

Main.o: Main.cpp
	g++ -c -fopenmp Main.cpp

CSR.o: CSR.cpp
	g++ -c -fopenmp CSR.cpp 

Ellpack.o: Ellpack.cpp
	g++ -c -fopenmp Ellpack.cpp
clean:
	rm -rf *.o 
	rm exec 
