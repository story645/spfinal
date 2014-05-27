#include <stdio.h>
#include "cpu_proximity.h"
#include "cpu_timer.h"

extern "C" void test_kdtree(float *data, unsigned int nSamples, unsigned int dim)
{
	Timer timer;
	timer.start();

	/**
	unsigned int nSamples = 10000;
	unsigned int dim = 20;
	unsigned int K = 10;
  	float* data = new float[nSamples * dim];
	unsigned int* KNNResult = new unsigned int[nSamples * K];
	for(unsigned int i = 0; i < nSamples * dim; ++i)
	{
		data[i] = rand() / (RAND_MAX + 1.0f);
	}
	**/
	unsigned int K = 10;
	unsigned int* KNNResult = new unsigned int[nSamples * K];
	
	proximityComputation_kdtree<20, float>(data, nSamples, data, nSamples, K, 0.0f, KNNResult);

	FILE* file = fopen("knn_kdtree.txt", "w");
	for(unsigned int i = 0; i < nSamples; ++i)
	{
		for(unsigned int j = 0; j < K; ++j)
		{
			fprintf(file, "%d ", KNNResult[j * nSamples + i]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
	
	/**
	delete [] data;
	data = NULL;
	**/
	delete [] KNNResult;
	KNNResult = NULL;

	timer.stop();
	printf("kdtree costs, %3f ms\n", timer.getElapsedTime());
}
