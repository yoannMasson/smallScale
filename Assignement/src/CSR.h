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
	std::vector<int> as;
	std::vector<int> ja;
	std::vector<double> a;

public:


	CSR(std::string filePath);
	virtual ~CSR();
};



#endif /* SRC_CSR_H_ */
