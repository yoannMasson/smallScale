all: Main.o CSR.o
	g++ Main.o -fopenmp CSR.o -o exec

Main.o: Main.cpp
	g++ -c -fopenmp Main.cpp

CSR.o: CSR.cpp
	g++ -c -fopenmp CSR.cpp 
