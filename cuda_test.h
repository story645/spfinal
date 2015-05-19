#ifndef __CUDA_TEST_H_
#define __CUDA_TEST_H_

void testall(float*, unsigned int, unsigned int, unsigned int, unsigned int, 
	     unsigned int, unsigned int, unsigned int);

void bruteforce(unsigned int, unsigned int, unsigned int, unsigned int, 
	float*, float*, unsigned int*, unsigned int*, 
	unsigned int*, unsigned int*, unsigned int);

void locsenhash(unsigned int, unsigned int, unsigned int, 
		unsigned int, unsigned int, unsigned int,
		float*, float*, unsigned int*, unsigned int*,
		unsigned int*, unsigned int*, unsigned int);
		
void printResults(char*, unsigned int*, unsigned int, unsigned int);

#endif
