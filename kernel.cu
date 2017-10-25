
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
using namespace std;
#define N 3 //rowsize
#define M 4 // columnsize
const int blockNUM = 4;
const int threadNUM =3;
void mxv(const int rowsize,const int columnsize,
         const float*matrix,const float*v,float*r)
         {
             for(int i=0;i<rowsize;i++)
             {
                 float re=0.0f;
                 for(int j=0;j<columnsize;j++)
                    re+=(matrix[i*columnsize+j]*v[j]);
                 r[i]=re;
             }
			 cout <<"CPU:";
             for(int i=0;i<rowsize;i++)
                cout << r[i]<< " ";
             cout <<endl;
         }

static void __global__ mxvNaive(int rowSize, int columnSize, int columnPitch,
const float *d_matrix, const float *d_vec, float *d_r)
  {
      int id = threadIdx.x+blockIdx.x*blockDim.x;
      if(id<rowSize)
      {
          float temp=0;
          for(int i=0;i<columnSize;i++)
          {
              temp+=d_matrix[id*columnPitch+i]*d_vec[i];
          }
          d_r[id]=temp;
      }
}
int main()
{
    float *matrix=(float*)malloc(N*M*sizeof(float));
	float *vec=(float*)malloc(M*sizeof(float));
	float *r =(float*)malloc(N*sizeof(float));
	float *dev_matrix,*dev_vec,*dev_r;
	cudaMalloc((void**)&dev_vec,M*sizeof(float));
	cudaMalloc((void**)&dev_matrix,M*N*sizeof(float));
	cudaMalloc((void**)&dev_r,N*sizeof(float));
	for(int i=1;i<=N*M;i++)
		matrix[i-1]=i;
		for(int i=0;i<M;i++)
            vec[i]=i+1;
	cudaMemcpy(dev_matrix,matrix,M*N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec,vec,M*sizeof(float),cudaMemcpyHostToDevice);
	mxvNaive<<<blockNUM,threadNUM>>> (N,M,M,dev_matrix,dev_vec,dev_r);
	cudaMemcpy(r,dev_r,N*sizeof(float),cudaMemcpyDeviceToHost);
	cout << "GPU:";
	for(int i=0;i<N;i++)
		cout <<r[i]<<" ";
	cout <<endl;
	mxv(N,M,matrix,vec,r);
    return 0;
}
