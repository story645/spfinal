#include <stdio.h>
#include "cpu_proximity.h"
#include "cpu_timer.h"

extern "C" void test_kdtree(float *data, unsigned int nSamples, unsigned int dim){
   Timer timer;
   timer.start();
   
   //printf("I used to work\n");

   unsigned int K = 5;
   unsigned int* KNNResult = new unsigned int[nSamples * K];
   //printf("I could allocate me\n");
   proximityComputation_kdtree<9, float>(data, nSamples, data, nSamples, K, 0.0f, KNNResult);
   //printf("I don't work anymore\n");
   
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
	
   delete [] KNNResult;
   KNNResult = NULL;
   
   timer.stop();
   printf("kdtree costs, %3f ms\n", timer.getElapsedTime());
}
