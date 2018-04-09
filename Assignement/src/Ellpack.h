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
	/*
	 Construct an Ellpack object using the path of a file containing a .mtx file
	 @param filePath, a string containing the path to a .mtx file
	 @throw Exception, if the file is not correctly formatted
	 */
	Ellpack(const std::string filePath);
	virtual ~Ellpack();
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
	 * Overwrite the << operator, to print an Ellpack object
	 */
	friend std::ostream& operator<<(std::ostream& os, Ellpack& obj);
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

};


#endif /* SRC_ELLPACK_H_ */
