#include<stdio.h>
#include<stdlib.h>
#define __USE_C99_MATH
#include<string.h>
#include <stdbool.h>
#include<math.h>
#include <mpi.h>

#include <stdbool.h>

bool test_case1();

void op3(float* a_vec, int m, float *b_vec, int ranks) ;


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

int compare (const void * a, const void * b)
{
  float A = *(const float*) a;
  float B = *(const float*) b;
  return (A > B) - (A < B);
}




int main(int argc, char *argv[]) {
     if(argc != 3) {
        printf("Usage: %s <Input_file> <Output_file>\n", argv[0]);
        return -1;
    }
    
    char Input_filename[500];
    char Output_filename[500];

    strcpy(Input_filename, argv[1]);
    strcpy(Output_filename, argv[2]);

    int rankid, size;

    long int total_input_size;
    int used_ranks, dimensions, num_dims, *dims;

    static float *Input, *output;
    float *out_loc, *Input_loc;
    double start[3],end[3],time[3];

    //Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);   
    
    if (rankid == 0)
    {
        start[0] = MPI_Wtime();
        //Read dimensions
        int *dimensions_original = read_dims(Input_filename);
        
        if(dimensions_original == NULL) {
            return -1;
        }

        num_dims = dimensions_original[0];
        dims = dimensions_original+1;

        //Read data
        //A
        Input = read_array(Input_filename, dims, num_dims);
        if(Input == NULL) {
            return -1;
        }
        //Set output
        dimensions = dims[0];
        total_input_size = dimensions;
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

        end[0] = MPI_Wtime(); // end time
    
	    time[0] = end[0] - start[0];
	    printf ("Rank: %d loaded all data in  = %f milliseconds \n" , rankid , 1000.*time[0]) ;

    }


    MPI_Barrier(MPI_COMM_WORLD); // Synchronisation with master rank
    start[1] = MPI_Wtime();



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
        
        // allocating space for output and A
        Input_loc = malloc(sizeof(float) * dimensions/used_ranks );
        
        int chunksize = dimensions/used_ranks;
        //scatter matrix A to active ranks and broadcast B
        MPI_Scatter(Input, chunksize, MPI_FLOAT, Input_loc,chunksize, MPI_FLOAT, 0, active_ranks);
        // Active ranks will run calculations
        qsort(Input_loc,chunksize,sizeof(float),compare);
        //sort(Input_loc, 0, chunksize-1);

        if(rankid == 0)
        {
        out_loc = malloc(sizeof(float) * dimensions );
        }
        //send data back to main rank
        MPI_Gather(Input_loc, chunksize, MPI_FLOAT, out_loc, chunksize, MPI_FLOAT, 0, active_ranks);

        if(rankid == 0)
        {
            if(used_ranks<2)
            {
                for (int i = 0; i < dimensions;i++)
                {
                    output[i] = out_loc[i];
                }

            }
            else
            {
            op3(out_loc, dimensions, output, used_ranks);
            }
        }
        
    }

	end[1] = MPI_Wtime(); // end time
    
	time[1] = end[1] - start[1] ;
    if (rankid < used_ranks)
        {
	printf ("rank %d finished with time = %f milliseconds\n" , rankid , 1000.*time[1]);
        }
    else
    {
    printf ("rank %d finished with time = %f milliseconds (This rank wasn't utilised for calculations)\n" , rankid , 1000.*time[1]);
    }

   if (rankid == 0)
    {  start[2] = MPI_Wtime();
     
        FILE *file = fopen(Output_filename,"w");

        if(file == NULL) {
            printf("Unable to open file: %s", Output_filename);
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

    end[2] = MPI_Wtime(); // end time
	time[2] = end[2] - start[2];
	printf (" rank %d finished writing data into file in %f milliseconds \n" , rankid , 1000.*time[2] ) ;
    
    }
    
    MPI_Finalize();
    
}

//OP3 takes received arrays that are all combined in one array and combine them together in merge sort fashion
void op3(float* a_vec, int m, float *b_vec, int ranks) 
{
    float (*a)[m/ranks] = (float(*)[m/ranks]) a_vec;
    float (*b)[m/ranks] = (float(*)[m/ranks]) b_vec;

    for (int i = 0; i <ranks/2;i++)
    {
        int k =0;
        int o = 0;
        for (int j = 0; j <m/ranks*2;j++)
        {
            //If one of the arrays have no number left, put the rest of the second array at the end
            if(k>=(m/ranks) || i*2+1 >= ranks)
            {
                b[i][j] = a[i*2][o];
                o++;
            }
            else if (o>=(m/ranks))
            {
                b[i][j] = a[i*2+1][k];
                k++;
            }
            else
            {
                // Compare even array with odd one and write lower number
                if(a[i*2][o] < a[i*2+1][k])
                {
                    b[i][j] = a[i*2][o];
                    o++;
                }
                else
                {
                    b[i][j] = a[i*2+1][k];
                    k++;
                }
            }
        }
    }
}