/*
 * Ellpack.h
 *
 *  Created on: 29 mars 2018
 *      Author: yoann
 */

#ifndef SRC_ELLPACK_H_
#define SRC_ELLPACK_H_
#include <string>
#include <vector>

class Ellpack {


private:

	int M;
	int N;
	int L;
	int MAXNZ;
	int** ja;
	double** as;

public:
	Ellpack(const std::string filePath);
	virtual ~Ellpack();
//	double serialVectorProduct(double* vector, double* solution);
//	double openMPVectorProduct(double* vector, double* solution);
	friend std::ostream& operator<<(std::ostream& os, Ellpack& obj);
//	int getM();
//	int getN();

};


#endif /* SRC_ELLPACK_H_ */
