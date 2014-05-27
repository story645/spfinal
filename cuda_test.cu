#include "cuda_proximity.h"
#include "cuda_timer.h"
#include <cudpp.h>
#include <cutil.h>
#include "reduce_kernel.h"
#include <time.h>
#include "cuda_test.h"

#define BS 1
#define LSH 1
#define RESULT 0

void testall(float *data, unsigned int nSamples, unsigned int dim)
{
	unsigned int nQueries = 20000;
	unsigned int K = 250;
	
	float* query = NULL;
	unsigned int* KNNResult = NULL;
	unsigned int* KNNResult_query = NULL;
	CPUMALLOC((void**)&data, sizeof(float) * nSamples * dim);
	CPUMALLOC((void**)&query, sizeof(float) * nQueries * dim);
	CPUMALLOC((void**)&KNNResult, sizeof(unsigned int) * nSamples * K);
	CPUMALLOC((void**)&KNNResult_query, sizeof(unsigned int) * nQueries * K);
	
	for(unsigned int i = 0; i < nSamples * dim; ++i)
	{
		data[i] = rand() / (RAND_MAX + 1.0f);
	}
	
	for(unsigned int i = 0; i < nQueries * dim; ++i)
	{
		query[i] = rand() / (RAND_MAX + 1.0f);
	}
	
	float* d_data = NULL;
	float* d_query = NULL;
	unsigned int* d_KNNResult = NULL;
	unsigned int* d_KNNResult_query = NULL;
	GPUMALLOC((void**)&d_data, sizeof(float) * nSamples * dim);
	GPUMALLOC((void**)&d_query, sizeof(float) * nQueries * dim);
	GPUMALLOC((void**)&d_KNNResult, sizeof(unsigned int) * nSamples * K);
	GPUMALLOC((void**)&d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
	
	TOGPU(d_data, data, sizeof(float) * nSamples * dim);
	TOGPU(d_query, query, sizeof(float) * nQueries * dim);


    	if(BS)
	{
		//data points self query using radixsort

		unsigned int timer3 = 0;
		startTimer(&timer3);
		
		proximityComputation_bruteforce2(d_data, nSamples, d_data, nSamples, dim, K, 0.0f, d_KNNResult);	
		FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
		endTimer("brute-force KNN -  data points self query- using radixsort", &timer3);	

		if(RESULT)
		{
			FILE* file3 = fopen("knn_bf2.txt", "w");
			for(unsigned int i = 0; i < nSamples; ++i)
			{
				for(unsigned int j = 0; j < K; ++j)
				{
					fprintf(file3, "%d ", KNNResult[j * nSamples + i]);
				}
				fprintf(file3, "\n");
			}
			fclose(file3);
		}
		
		//separate data/query points using radixsort		
		unsigned int timer4 = 0;
		startTimer(&timer4);
		
		proximityComputation_bruteforce2(d_data, nSamples, d_query, nQueries, dim, K, 0.0f, d_KNNResult_query);
		FROMGPU(KNNResult_query, d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
		endTimer("brute-force KNN - separate data/query points - using radixsort", &timer4);
		
		if(RESULT)
		{
			FILE* file4 = fopen("knn_query_bf2.txt", "w");
			for(unsigned int i = 0; i < nQueries; ++i)
			{
				for(unsigned int j = 0; j < K; ++j)
				{
					fprintf(file4, "%d ", KNNResult_query[j * nQueries + i]);
				}
				fprintf(file4, "\n");
			}
			fclose(file4);
		}		
	}

	if(LSH )
	{
		float* h_lower = NULL;
		float* h_upper = NULL;
		CPUMALLOC((void**)&h_lower, sizeof(float) * dim);
		CPUMALLOC((void**)&h_upper, sizeof(float) * dim);
		
		for(unsigned int i = 0; i < dim; ++i)
		{
			h_upper[i] = 1;
			h_lower[i] = 0;
		}
		
		int LSH_L = 5;

		//data points self query
		unsigned int timer1 = 0;
		startTimer(&timer1);
		
		proximityComputation_LSH(d_data, nSamples, d_data, nSamples, dim, K, LSH_L, 0.0f, h_upper, h_lower, d_KNNResult);
		FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
		endTimer("LSH KNN -data point self query", &timer1);
		
		if(RESULT)
		{
			FILE* file1 = fopen("knn_lsh.txt", "w");
			for(unsigned int i = 0; i < nSamples; ++i)
			{
				for(unsigned int j = 0; j < K; ++j)
				{
					fprintf(file1, "%d ", KNNResult[j * nSamples + i]);
				}
				fprintf(file1, "\n");
			}
			fclose(file1);
		}
	
	
		unsigned int timer2 = 0;
		startTimer(&timer2);
		proximityComputation_LSH(d_data, nSamples, d_query, nQueries, dim, K, LSH_L, 0.0f, h_upper, h_lower, d_KNNResult_query);
		FROMGPU(KNNResult_query, d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
		endTimer("LSH KNN - separte data/query points", &timer2);

		if(RESULT)
		{
			FILE* file2 = fopen("knn_query_lsh.txt", "w");
			for(unsigned int i = 0; i < nQueries; ++i)
			{
				for(unsigned int j = 0; j < K; ++j)
				{
					fprintf(file2, "%d ", KNNResult_query[j * nQueries + i]);
				}
				fprintf(file2, "\n");
			}
			fclose(file2);
		}		
		CPUFREE(h_lower);
		CPUFREE(h_upper);		
	}
	
	GPUFREE(d_data);
	GPUFREE(d_KNNResult);
	CPUFREE(data);
	CPUFREE(KNNResult);
	
	GPUFREE(d_query);
	GPUFREE(d_KNNResult_query);
	CPUFREE(query);
	CPUFREE(KNNResult_query);
	
}
