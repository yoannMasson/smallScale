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

	/*
		 Construct a CSR object using the path of a file containing a .mtx file
		 @param filePath, a string containing the path to a .mtx file
		 @throw Exception, if the file is not correctly formatted
	*/
	CSR(std::string filePath);
	/*
		 * Compute the serial version of the matrix vector product
		 * @param vector, a pointer to the vector of the product
		 * @param solution, a pointer to the resulting vector, will be filled during the method
		 * @return the time in second
		 */
	double serialVectorProduct(double* vector, double* solution);
	/*
		 * Compute the openMP version of the matrix vector product
		 * @param vector, a pointer to the vector of the product
		 * @param solution, a pointer to the resulting vector, will be filled during the method
		 * @param nCore, the number of core to be used in the parallelization
		 * @return the time in second
		 */
	double openMPVectorProduct(double* vector, double* solution,int nCore);
	/*
		 * Compute the cuda version of the matrix vector product
		 * @param vector, a pointer to the vector of the product
		 * @param solution, a pointer to the resulting vector, will be filled during the method
		 * @return the time in second
		 */
	double cudaVectorProduct(double* vector, double* solution);
	/*
	 * Overwrite the << operator, to print a CSR object
	 */
	friend std::ostream& operator<<(std::ostream& os, CSR& obj);
	/*
	 * Return the number of row of the original matrice
	 * @return M
	 */
	int getM();
	/*
	 * Return the number of column of the original matrice
	 * @return N
	 */
	int getN();
	/*
	 * Return the number of non null entry of the original matrice
	 * @return L
	 */
	int getL();
	/*
	 * Return a pointer to the array representing the IRP array
	 * @return IRP
	 */
	int* getIRP();
	/*
	 * Return a pointer to the array representing the JA array
	 * @return JA
	 */
	int* getJA();
	/*
	 * Return a pointer to the double-dimension array representing the IRP array
	 * @return AS
	 */
	double* getAS();
	virtual ~CSR();
};



#endif /* SRC_CSR_H_ */
