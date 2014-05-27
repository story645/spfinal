#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>

#include "cuda_test.h"
#include "cpu_test.h"
#include "cuda_defs.h"

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

int main()
{
	unsigned int nSamples = 10000;
        unsigned int dim = 20;
	
	unsigned int t = 1;
	unsigned int lat = 190;
	unsigned int lon = 384;
	unsigned int start = 1985;
	unsigned int end = 2009;
	unsigned int time = ((end-start) + 1)*12; 
	
	char *filename = "../data/TMP_198501_a.nc";
	
	//netCDF fileit and data var
	int ncid, varid;
	char *varname = "TMP_2maboveground";
	//loop inds and error_handling
	int ld, la, ln, retval;
	
	//contains data_in
	float *data_in, *data;	
	
	/** Open the file. NC_NOWRITE tells netCDF 
  	we want read-only access  to the file.*/
  	if((retval = nc_open(filename, NC_NOWRITE, &ncid)))		
		ERR(retval);		
	
	/** Get the varid of the data variable, 
 	    based on its name. */
   	if ((retval = nc_inq_varid(ncid, varname, &varid)))
		ERR(retval);


	
	
  	/*  Allocate enough space.. */	
  	data_in = (float *)malloc(sizeof(float)*t*lat*lon);
	
	/* Read the data. */	
   	if ((retval = nc_get_var_float(ncid, varid, data_in)))
      	ERR(retval);

   	/* print out data for sanity check */
	ld = 0;
	for(la=0; la<lat; la++){
		for (ln=0; ln<lon; ln++){
			printf("[%03d, %03d] = %f\n", la, ln, data_in[ld]);
			ld++;
		}
	}

   	/* Close the file, freeing all resources. */
   	if ((retval = nc_close(ncid)))
     		 ERR(retval);

   	printf("*** SUCCESS reading %s!\n", filename);

	
	//testall(data, nSamples, dim);
	//test_kdtree(data, nSamples, dim);

	
}

