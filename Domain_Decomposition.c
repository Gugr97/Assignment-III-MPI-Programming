#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

int main(int argc, char *argv[]){

    bool blocked = true;

    int rank, size, i, provided;
    
    // number of cells (global)
    int nxc = 128; // make sure nxc is divisible by size
    double L = 2*3.1415; // Length of the domain
    

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // number of nodes (local to the process): 0 and nxn_loc-1 are ghost cells 
    int nxn_loc = nxc/size + 3; // number of nodes is number cells + 1; we add also 2 ghost cells
    double L_loc = L/((double) size);
    double dx = L / ((double) nxc);
    
    // define out function
    double *f = calloc(nxn_loc, sizeof(double)); // allocate and fill with z
    double *dfdx = calloc(nxn_loc, sizeof(double)); // allocate and fill with z

    for (i=1; i<(nxn_loc-1); i++)
      f[i] = sin(L_loc*rank + (i-1) * dx);
    
    // need to communicate and fill ghost cells f[0] and f[nxn_loc-1]
    // communicate ghost cells

    // send ghost cells from neighboring processes
    // communicate ghost cells

    if (blocked)
    {
        if (rank == 0) {
            MPI_Sendrecv(&f[nxn_loc-3], 1, MPI_DOUBLE, rank+1, 0, &f[0], 1, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&f[2], 1, MPI_DOUBLE, size-1, 0, &f[nxn_loc-1], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        } else if (rank == size-1) {
            MPI_Sendrecv(&f[nxn_loc-3], 1, MPI_DOUBLE, 0, 0, &f[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&f[2], 1, MPI_DOUBLE, rank-1, 0, &f[nxn_loc-1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
        } else {
            MPI_Sendrecv(&f[nxn_loc-3], 1, MPI_DOUBLE, rank+1, 0, &f[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&f[2], 1, MPI_DOUBLE, rank-1, 0, &f[nxn_loc-1], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    else{
        MPI_Request request[4];
        MPI_Status status[4];

        if (rank == 0) {
            MPI_Irecv(&f[0], 1, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Isend(&f[nxn_loc-3], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request[1]);
            MPI_Irecv(&f[nxn_loc-1], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request[2]);
            MPI_Isend(&f[2], 1, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD, &request[3]);
        } else if (rank == size-1) {
            MPI_Irecv(&f[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Isend(&f[nxn_loc-3], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[1]);
            MPI_Irecv(&f[nxn_loc-1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[2]);
            MPI_Isend(&f[2], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request[3]);
        } else {
            MPI_Irecv(&f[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Isend(&f[nxn_loc-3], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request[1]);
            MPI_Irecv(&f[nxn_loc-1], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request[2]);
            MPI_Isend(&f[2], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request[3]);
        }

        MPI_Waitall(4, request, status);

    }

    // here we finish the calculations

    // calculate first order derivative using central difference
    // here we need to correct value of the ghost cells!
    for (i=1; i<(nxn_loc-1); i++)
      dfdx[i] = (f[i+1] - f[i-1])/(2*dx);

    
    // Print f values
    if (rank==0){ // print only rank 0 for convenience
        printf("My rank %d of %d\n", rank, size );
        printf("Here are my values for dfdx including ghost cells:\n");
        for (i=0; i<nxn_loc; i++)
        {
            double x = L_loc * rank + (i-1) * dx;
	        printf("%s\t%f\t%s\t%f\n","dfdx:", dfdx[i], "cos(x):", cos(x));
            printf("\n");
        }
        printf("Here are my values for f:\n");
        for (i=0; i<nxn_loc; i++)
        {
            double x = L_loc * rank + (i-1) * dx;
	        printf("%s\t%f\t%s\t%f\n","f:", f[i], "sin(x):", sin(x));
            printf("\n");
        }
    }
    MPI_Finalize();
}