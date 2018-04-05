/*
 * CSR.h
 *
 *  Created on: 14 mars 2018
 *      Author: yoann
 */

#ifndef SRC_CSR_H_
#define SRC_CSR_H_
#include <string>
#include <vector>

class CSR {

private:

	int M;
	int N;
	int L;
	int *irp;
	int *ja;
	double *as;

public:


	CSR(std::string filePath);
	double serialVectorProduct(double* vector, double* solution);
	double openMPVectorProduct(double* vector, double* solution);
	double cudaVectorProduct(double* vector, double* solution);
	friend std::ostream& operator<<(std::ostream& os, CSR& obj);
	int getM();
	int getN();
	int getL();
	int* getIRP();
	int* getJA();
	double* getAS();
	virtual ~CSR();
};



#endif /* SRC_CSR_H_ */
