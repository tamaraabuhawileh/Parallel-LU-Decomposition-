%%writefile snippet2.cu
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <time.h>
using namespace std;




//LU Decompistion


__global__ void lower_for (float** a ,float** l,float** u, int size ,int i )
{
    // Lower triangular matrix
int k = blockIdx.x * blockDim.x + (threadIdx.x + i);


if (k < size)
        {
            if (i == k)
            {
                l[i][i] = 1.0; // Diagonal elements of L are 1
            }
            
            else
            {
                float sum = 0.0;
                for (int j = 0; j < i; j++)
                {
                    sum += l[k][j] * u[j][i];
                }
                l[k][i] = (a[k][i] - sum) / u[i][i];
            }
        }

}

__global__ void upper_for (float** a ,float** l,float** u, int size ,int i )
{
    // Upper triangular matrix
int k = blockIdx.x * blockDim.x + (threadIdx.x + i) ;


if (k < size)
        {
          float sum = 0.0;
            for (int j = 0; j < i; j++)
            {
                sum += l[i][j] * u[j][k];
            }
            u[i][k] = a[i][k] - sum;

        }

}






//print the matrix out
void print_matrix(float** matrix, int size)
{
    //for each row...
    for (int i = 0; i < size; i++)
    {
        //for each column
        for (int j = 0; j < size; j++)
        {
            //print out the cell
            cout << left << setw(9) << setprecision(4) << matrix[i][j] << left <<  setw(10);
        }
        //new line when ever row is done
        cout << endl;
    }
}





//fill the array with random values (done for a)
void random_fill(float** matrix, int size)
{
    //fill a with random values
    cout << "Producing random values " << endl;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = ((rand()%10)+1) ;
        }
    }

    //Ensure the matrix is diagonal dominant to guarantee invertible-ness
    //diagCount well help keep track of which column the diagonal is in
    int diagCount = 0;
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            //Sum all column vaalues
            sum += abs(matrix[i][j]);
        }
        //Remove the diagonal  value from the sum
        sum -= abs(matrix[i][diagCount]);
        //Add a random value to the sum and place in diagonal position
        matrix[i][diagCount] = sum + ((rand()%5)+1);
        ++diagCount;
        sum = 0;
    }
}
  //allocate vectors on unified memory **

 void initialize_matrices(float** a, float** l, float** u, int size)
  {



  for(int i=0; i<size; i++){
    cudaMallocManaged(&a[i], size * sizeof(float));
    cudaMallocManaged(&l[i], size * sizeof(float));
    cudaMallocManaged(&u[i], size * sizeof(float));

  }
}




int main(){
  int n=5;

  float **a, **l, **u;
   int i =0;

  srand(1);
  cudaMallocManaged(&a, n * sizeof(float*));
  cudaMallocManaged(&l, n * sizeof(float*));
  cudaMallocManaged(&u, n * sizeof(float*));
initialize_matrices(a,l,u,n);
random_fill(a, n);


// add the clock
 double runtime;
  runtime = clock()/(double)CLOCKS_PER_SEC;
for( i=0; i<n; ++i){

    lower_for <<< 16,250 >>> (a , l, u, n , i );
    cudaDeviceSynchronize();
    upper_for <<< 16,250 >>> ( a , l, u, n , i );
    cudaDeviceSynchronize();

   }

runtime = clock() - runtime;
cout << "Runtime for LU Decomposition is: " << (runtime)/(double)(CLOCKS_PER_SEC) << endl;







// print the matrices
cout << "A Matrix: " << endl;
print_matrix(a, n);
cout << "L Matrix: " << endl;
print_matrix(l, n);

cout << "U Matrix: " << endl;
print_matrix(u, n);

cout << "Runtime for LU Decomposition is: " << (runtime)/float(CLOCKS_PER_SEC) << endl;


for(i=0; i<n; i++){
    cudaFree(a[i]);
     cudaFree(l[i]);
     cudaFree(u[i]);
  }
   cudaFree(a);
     cudaFree(l);
     cudaFree(u);

  return 0;
}