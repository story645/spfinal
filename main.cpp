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
	
	unsigned int lat = 190;
	unsigned int lon = 384;
	unsigned int start = 1985;
	unsigned int end = 2009;
	unsigned int time = ((end-start) + 1)*12; 
	
	char *filename = "../data/TMP_198501_a.nc";
	
	//netCDF fileit and data var
	int ncid, varid;
	
	char *varname = "TMP_2maboveground";
	int data_in[lat][lon];
	//loop inds and error_handling
	int x, y, retval;
		
	
	/** Open the file. NC_NOWRITE tells netCDF 
  	we want read-only access  to the file.*/
  	if((retval = nc_open(filename, NC_NOWRITE, &ncid)))		ERR(retval);		
	
	/** Get the varid of the data variable, 
 	    based on its name. */
   	if ((retval = nc_inq_varid(ncid, varname, &varid))){
      		//ERR(retval);
   		printf("var name: %d\n", varid);
		ERR(retval);
	}	
	/* Read the data. */	
   	if ((retval = nc_get_var_int(ncid, varid, &data_in[0][0])))
      	ERR(retval);

   	/* Check the data. */
  	 for (x = 0; x < lat; x++)
      		for (y = 0; y < lon; y++)
	 		printf("%f", data_in[x,y]);

   	/* Close the file, freeing all resources. */
   	if ((retval = nc_close(ncid)))
     		 ERR(retval);

   	printf("*** SUCCESS reading %s!\n", filename);

	
	//testall(data, nSamples, dim);
	//test_kdtree(data, nSamples, dim);

	
}

