#include<stdio.h>
#include<stdlib.h>
#define __USE_C99_MATH
#include<string.h>
#include <stdbool.h>
#include<math.h>
#include<mpi.h>

void op1(float*, int, int, float*, int, float*, int, int, int, int);

long int product(int *array, int n) {
    long int product = 1;
    for(int i=0; i<n; i++) {
        product *= array[i];
    }
    return product;
}

int *read_dims(char *filename) {
    FILE *file = fopen(filename,"r");
    
    if(file == NULL) {
        printf("Unable to open file: %s", filename);
        return NULL;
    }

    char firstline[500];
    fgets(firstline, 500, file);
    
    int line_length = strlen(firstline);

    int num_dims = 0;
    for(int i=0; i<line_length; i++) {
        if(firstline[i] == ' ') {
            num_dims++;
        }
    }
    
    int *dims = malloc((num_dims+1)*sizeof(int));
    dims[0] = num_dims;
    const char s[2] = " ";
    char *token;
    token = strtok(firstline, s);
    int i = 0;
    while( token != NULL ) {
        dims[i+1] = atoi(token);
        i++;
        token = strtok(NULL, s);
    }
    fclose(file);
    return dims;
}

float * read_array(char *filename, int *dims, int num_dims) {
    FILE *file = fopen(filename,"r");

    if(file == NULL) {
        printf("Unable to open file: %s", filename);
        return NULL;
    }

    char firstline[500];
    fgets(firstline, 500, file);

    //Ignore first line and move on since first line contains 
    //header information and we already have that. 

    long int total_elements = product(dims, num_dims);

    float *one_d = malloc(sizeof(float) * total_elements);

    for(int i=0; i<total_elements; i++) {
        fscanf(file, "%f", &one_d[i]);
    }
    fclose(file);
    return one_d;
}



int main(int argc, char *argv[]) {
    if(argc != 4) {
        printf("Usage: %s <filename_input> <filename_kernel> <filename_expected_output>\n", argv[0]);
        return -1;
    }

    bool match = true;

    char input_filename[500];
    char kernel_filename[500];
    char output_filename[500];

    strcpy(input_filename, argv[1]);
    strcpy(kernel_filename, argv[2]);
    strcpy(output_filename, argv[3]);

    int rankid, size;

    long int total_input_size;
    int *input_dims, *kernel_dims, input_num_dims, kernel_num_dims, offsetA, range;

    static float *input_data, *kernel_data, *output;
    float *out_loc;

    //Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rankid == 0)
    {  double start = MPI_Wtime();

        //Read dimensions from Input
        int *input_dims_original = read_dims(input_filename);
        if(input_dims_original == NULL) {
            return -1;
        }
        input_num_dims = input_dims_original[0];
        input_dims = input_dims_original+1;

        //Read data from Input
        input_data = read_array(input_filename, input_dims, input_num_dims);
        if(input_data == NULL) {
            return -1;
        }

        //Read dimensions from Kernel
        int *kernel_dims_original = read_dims(kernel_filename);
        if(kernel_dims_original == NULL) {
            return -1;
        }
        kernel_num_dims = kernel_dims_original[0];
        kernel_dims = kernel_dims_original+1;

        //Read data from Kernel
        kernel_data = read_array(kernel_filename, kernel_dims, kernel_num_dims);
        if(kernel_data == NULL) {
            return -1;
        }

        total_input_size = product(input_dims, input_num_dims);
        output = malloc(sizeof(float) * total_input_size);

        double end = MPI_Wtime(); // end time
    
	    double time = end - start ;
	    printf ("Rank: %d loaded all data in  = %f milliseconds \n" , rankid , 1000.*time ) ;


    }  

    MPI_Barrier(MPI_COMM_WORLD); // Synchronisation with master rank
    double start = MPI_Wtime();
    


 
    // Broadcast dimension
    MPI_Bcast ( &input_num_dims, 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast ( &kernel_num_dims, 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast ( &total_input_size, 1 , MPI_LONG , 0 , MPI_COMM_WORLD);
    
    if (rankid > 0 ) {
    input_dims = malloc(sizeof(float) * 3);
    kernel_dims = malloc(sizeof(float) * 2);
    }
    
    MPI_Bcast ( input_dims, 3 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast ( kernel_dims, 2 , MPI_INT , 0 , MPI_COMM_WORLD);
    
  
    //Allocate space for input data, kernel data amd local output
        if (rankid > 0 ) {
    input_data = malloc(sizeof(float) * total_input_size);
    kernel_data = (float*) malloc( kernel_dims[0] * kernel_dims[0] * sizeof( float* ));
    }
    out_loc = calloc(total_input_size, sizeof(float));



    //Broadcast Input and Kernel
    MPI_Bcast (input_data, total_input_size , MPI_FLOAT , 0 , MPI_COMM_WORLD);
    MPI_Bcast (kernel_data, kernel_dims[0] * kernel_dims[0] , MPI_FLOAT , 0 , MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    //Decide which chunk should be calculated by this rank
    offsetA = (input_dims[0] / size)*rankid;
    
    if(input_dims[0] % size != 0)
    {
        range=(input_dims[0] / size)+1;

        if(rankid!=0)
        {
            
            offsetA=offsetA+1;
            
        }
    }
    else
    {
        range =(input_dims[0] / size);

    }


    op1(input_data, input_dims[1], input_dims[2], kernel_data, kernel_dims[0], out_loc, input_dims[0], offsetA, range, rankid);


    MPI_Reduce(out_loc , output , total_input_size , MPI_FLOAT , MPI_SUM , 0 , MPI_COMM_WORLD);


    free(input_data);
    free(kernel_data);

	double end = MPI_Wtime(); // end time
    
	double time = end - start ;
	printf (" rank %d finished in %f milliseconds \n" , rankid , 1000.*time ) ;



    if (rankid == 0)
    {   double start = MPI_Wtime();

       
        FILE *file = fopen(output_filename,"w");

        if(file == NULL) {
            printf("Unable to open file: %s", output_filename);
            return -1;
        }

        if (file != NULL) {
            for(int i=0; i<input_num_dims; i++) {
                fprintf(file, "%d ", input_dims[i]);
            }
            fprintf(file, "\n");
        }

        for(int i=0; i<total_input_size; i++) {
            fprintf(file, "%.6f ", output[i]);
        }

        fclose(file);
    
    free(output);
    double end = MPI_Wtime(); // end time
    
	double time = end - start ;
	printf (" rank %d finished writing data into file in %f milliseconds \n" , rankid , 1000.*time ) ;
    }
    MPI_Barrier(MPI_COMM_WORLD); 
    return !match;
    MPI_Finalize();
}


void op1(float *input_vec,  int m,  int n, float *filter_vec, int k, float *output_vec, int b, int start, int size, int rank)
{
    float(*input)[m][n] = (float(*)[m][n]) input_vec;
    float(*filter)[k] = (float(*)[k]) filter_vec;
    float(*output)[m][n] = (float(*)[m][n]) output_vec;

int offset1 = 0;
int offset2 = 0;
float temp;
// if size of the filter is odd
if (k  % 2 != 0)
{
offset1 = (k  / 2);
offset2 = (k  / 2);
}
// if size of the filter is even
else
{
offset1 = (k  / 2)-1;
offset2 = (k  / 2);
}

int y,x,i,z,max;

if(start+size > b || size == 1)
{
max = b;
}
else
{
max = start+size;
}

// loop for b - number of batches
   for (z = start; z < max; ++z)
    {
        //loop to iterate through m
        for (y = 0; y < m; ++y)
        {;
            //loop to iterate through n
            for (x = 0; x < n; ++x)
            { 
                //if statement to ensure that filter is not applied to values in first and last column/row
                if (x>=(offset1) && y>=(offset1) && x<(n-offset2) && y<(m-offset2))
                {

                    //iteration through the filter
                    for (i = 0; i < (k*k); i++)
                    {   
                        // s is equal to line on which we want to work                
                        int s = (i/k);
                        //calculate values multipled by filter and add them to the temp variable
                        temp+= (input[z][y-offset1+s][x-offset1+i-(k*s)] * filter[s][i-(k*s)]);
                    }
                    //move variables from temp and flush temp variable
                    output[z][y][x]=temp;
                    temp = 0;
                }
                else
                {
                //else just copy values from input;
                output[z][y][x] = input[z][y][x];
                }
                
                
            }
        }
        
    }
return;
}





