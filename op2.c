#include<stdio.h>
#include<stdlib.h>
#define __USE_C99_MATH
#include<string.h>
#include <stdbool.h>
#include<math.h>
#include <mpi.h>

#include <stdbool.h>

bool test_case1();

void op2(float* a_vec, int m, int n, float *b_vec, float *c_vec);


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

long int product(int *array, int n) {
    long int product = 1;
    for(int i=0; i<n; i++) {
        product *= array[i];
    }
    return product;
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
        printf("Usage: %s <filename_A> <filename_B> <filename_C>\n", argv[0]);
        return -1;
    }

    bool match = true;

    char A_filename[500];
    char B_filename[500];
    char C_filename[500];

    strcpy(A_filename, argv[1]);
    strcpy(B_filename, argv[2]);
    strcpy(C_filename, argv[3]);

    int rankid, size;

    long int total_input_size;
    int used_ranks, dimensions, num_dims, *dims;

    static float *A, *B, *output;
    float *out_loc, *A_loc; // Matrices




    //Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);   

    if (rankid == 0)
    {double start = MPI_Wtime();
        //Read dimensions
        int *dimensions_original = read_dims(A_filename);
        
        if(dimensions_original == NULL) {
            return -1;
        }

        num_dims = dimensions_original[0];
        dims = dimensions_original+1;

        //Read data
        //A
        A = read_array(A_filename, dims, num_dims);
        if(A == NULL) {
            return -1;
        }
        //B
        B = read_array(B_filename, dims, num_dims);
        if(B == NULL) {
            return -1;
        }
        //Set output
        dimensions = dims[0];
        total_input_size = dimensions * dimensions;
        output = malloc(sizeof(float) * total_input_size);

        //decide how many ranks should be utilised
        if(dimensions % size != 0)
        {
            for(int i = size - 1; i > 1; i--)
            {
                if (dimensions % i == 0)
                {
                  used_ranks = i;
                  break;  
                }

            }

        }
        else
        {
            used_ranks = size;
        }
        double end = MPI_Wtime(); // end time
    
	    double time = end - start ;
	    printf ("Rank: %d loaded all data in  = %f milliseconds \n" , rankid , 1000.*time ) ;
    }


    MPI_Barrier(MPI_COMM_WORLD); // Synchronisation with master rank
    double start = MPI_Wtime();

    // Broadcast dimension
    MPI_Bcast ( &dimensions, 1 , MPI_INT , 0 , MPI_COMM_WORLD);

    // Broadcast how many ranks should be utilised
    MPI_Bcast ( &used_ranks, 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    
    //Assign correct ranks to new communication group
    MPI_Comm active_ranks;
    int color = (rankid < used_ranks) ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rankid, &active_ranks);

    //continue communication only to active ranks
    if (rankid < used_ranks)
    {
        
        // Allocate space for B before receiving data
        if (rankid > 0) {
        B = malloc(sizeof(float) * dimensions * dimensions);          
        }

        // allocating space for output and A
        out_loc = malloc(sizeof(float) * dimensions * dimensions);
        A_loc = malloc(sizeof(float) * dimensions * dimensions/used_ranks );

        //scatter matrix A to active ranks and broadcast B
        MPI_Scatter(A, dimensions * dimensions/used_ranks   , MPI_FLOAT, A_loc, dimensions * dimensions/used_ranks, MPI_FLOAT, 0, active_ranks);
        MPI_Bcast (B, dimensions*dimensions, MPI_FLOAT, 0, active_ranks);

        // Active ranks will run calculations
        op2(A_loc, dimensions / used_ranks, dimensions, B, out_loc);

        // rank 0 will collect all data
        MPI_Gather(out_loc, dimensions*dimensions/used_ranks, MPI_FLOAT, output, dimensions*dimensions/used_ranks, MPI_FLOAT, 0, active_ranks);

        free(out_loc);
        free(A_loc);
    }

	double end = MPI_Wtime(); // end time
    
	double time = end - start ;
    if (rankid < used_ranks)
        {
	printf ("rank %d finished with time = %f milliseconds\n" , rankid , 1000.*time);
        }
    else
    {
    printf ("rank %d finished with time = %f milliseconds (This rank wasn't utilised for calculations)\n" , rankid , 1000.*time);
    }

    if (rankid == 0)
    {  double start = MPI_Wtime();
        
        FILE *file = fopen(C_filename,"w");

        if(file == NULL) {
            printf("Unable to open file: %s", C_filename);
            return -1;
        }

        if (file != NULL) {
            for(int i=0; i<num_dims; i++) {
                fprintf(file, "%d ", dims[i]);
            }
            fprintf(file, "\n");
        }

        for(int i=0; i<total_input_size; i++) {
            fprintf(file, "%.6f ", output[i]);
        }

        fclose(file);

    free(A);
    free(B);
    free(output);
    double end = MPI_Wtime(); // end time
    
	double time = end - start ;
	printf (" rank %d finished writing data into file in %f milliseconds \n" , rankid , 1000.*time ) ;
    }
    MPI_Barrier(MPI_COMM_WORLD); 
    return !match;
    MPI_Finalize();
    
}


void op2(float* a_vec, int m, int n, float *b_vec, float *c_vec) 
{
    float (*a)[n] = (float(*)[n]) a_vec;
    float (*b)[n] = (float(*)[n]) b_vec;
    float (*c)[n] = (float(*)[n]) c_vec;

float *temp = calloc(1, sizeof(float));

for (int i = 0; i <m;i++)
{
    for (int j = 0; j <n;j++)
    {
    
        for (int k = 0; k <n;k++)
        {
        *temp += a[i][k] * b[k][j];
        }        
        c[i][j]=*temp;
        *temp=0;
    }
}
free(temp);
}