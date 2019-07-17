#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#include <assert.h>


#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))
#define ARRAY_DEBUG 0
#define PREFIX_DEBUG 1
#define HIST_DEBUG 0
#define START_BIT_LOC 0
#define ERROR_CHECK 1
#define MAX_THREAD_COUNT 2097121
#define FORCE_POOLING 1
#define gpuErrchk(ans, KERN_NAME) { gpuAssert((ans), KERN_NAME, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char * kernelName, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert(%s): %s %s %d\n", kernelName, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//data generator
void dataGenerator(int* data, int count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}



//list of obvious optimizations to make:
/*
    convert the histogram pooling and reorder pooling code to use thread strides, so as to enable coalesced mem accesses
    switch the prefix sum to a non-naive algorithm
    if possible, upgrade prefix sum to use shuffle warp instructions
    for the histogram, because the expected size is at most 1024 bins, we should be able to get away with a shared mem histogram
*/


/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

//define the histogram kernel here
//kernel config based on r_h size
__global__ void histogram(int POOL_SIZE, int* i_r_h, int i_rh_size, int i_numbits ,int* o_histogram)
{
    int k;
    if(POOL_SIZE > 0)
    {
        k = (blockDim.x * blockIdx.x + threadIdx.x) * POOL_SIZE;
        int kindex;
        int kmax = k+POOL_SIZE;
        for(kindex = k; kindex < kmax; kindex++)
        {
            if(kindex < i_rh_size)
            {

                int h = bfe(i_r_h[kindex], START_BIT_LOC, i_numbits);    
                atomicAdd(&o_histogram[h], 1);
            }
            
        }
    }
    else
    {
        k = blockDim.x * blockIdx.x + threadIdx.x;
        if(k < i_rh_size)
        {
            int h = bfe(i_r_h[k], START_BIT_LOC, i_numbits);      //i assume start value is 0...?
                                                                        //nope... it's 32 i think
            atomicAdd(&o_histogram[h], 1);
        }    
    }
}

//notice how the num partitions will be the size of 2 to 1024... hmmm it seems they know arbitrary size prefix scan is tricky to get right
//because coincidently, a naive prefix scan can maybe work up to 1024
//this also might be a huge hint in itself as to how we go about this
//the idea then, is that we only work from 1 block... the block with the size of 2^numberOfBits... i think.... are numPartitions = 2^numberOfBits?
//define the prefix scan kernel here
//implement it yourself or borrow the code from CUDA samples
//so it seems the width is based on numPartitions
//remember that this is exclusive scan

//bug: oddly enough, prefix scan fails at partition size 1024 only... i wonder why

__global__ void prefixScan(int* i_histogram, int n, int* o_prefix_sum)
{
    extern __shared__ int temp[];

    int tid = threadIdx.x;  //we only use 1 block, so threadidx is good enough
    int offset = 1;
    if(tid < n)
    {
        if(tid > 0)
            temp[tid] = i_histogram[tid-1];
        else
            temp[tid] = 0;
    
        
        __syncthreads();


        for(offset = 1; offset < n; offset *= 2)
        {
            int lh = tid - offset;
            int rh = tid;
            if(lh > 0)
                temp[rh] += temp[lh];
            else
                temp[rh] = temp[rh];

            __syncthreads();
        }

        //write output
        o_prefix_sum[tid] = temp[tid];
        
    }
}

//define the reorder kernel here
__global__ void Reorder(int POOL_SIZE, int* i_r_h, int i_rh_size, int i_numbits, int* i_prefix_sum, int* o_r_h)
{

    int k;
    if(POOL_SIZE > 0)
    {
        k = (blockDim.x * blockIdx.x + threadIdx.x) * POOL_SIZE;    
        int kindex;
        int kmax = k+POOL_SIZE;
        for(kindex = k; kindex < kmax; kindex++)
        {
            if(kindex < i_rh_size )
            {
                int kval = i_r_h[kindex];
                int h = bfe(kval, START_BIT_LOC, i_numbits);     //i assume start value is 32 here as well, if we're using the same logic from histogram kernel
                int offset = atomicAdd(&i_prefix_sum[h], 1);
                o_r_h[offset] = kval;    
            }
        }
    }
    else
    {
        k = blockDim.x * blockIdx.x + threadIdx.x;    
        if(k < i_rh_size)
        {
            int kval = i_r_h[k];
            int h = bfe(kval, START_BIT_LOC, i_numbits);     //i assume start value is 32 here as well, if we're using the same logic from histogram kernel
            int offset = atomicAdd(&i_prefix_sum[h], 1);
            o_r_h[offset] = kval;
        }
    }
}


int isNumber(char * arg)
{
    int n;
    char ch;
    
    return sscanf(arg, "%d%c", &n, &ch) == 1;

}


//todo: add timing code
int main(int argc, char *argv[])
{
    if(argc != 3 || !isNumber(argv[1])  || !isNumber(argv[2]))
    {
        printf("ERROR: Must specify 2 arguments after ./proj3! parameters allowed are (in order):\n num_elements(int), num_partitions(int)\n");
        return 1;
    }


    int rSize = atoi(argv[1]);
    int numPartitions = atoi(argv[2]);
    
    int* r_h;

    gpuErrchk(cudaMallocHost((void**)&r_h, sizeof(int)*rSize),"r_h malloc stage"); //use pinned memory 
    printf("Generating data...\n");
    dataGenerator(r_h, rSize, 0, 1);
    printf("Finished generating data!\n");
    /* your code */
    //i will start measuring time from my code specifically
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start), "timing");
    gpuErrchk(cudaEventCreate(&stop), "timing");
    gpuErrchk(cudaEventRecord(start, 0), "timing");


    //allocate histogram, prefix sum, and reordered buffer
    int* h_histogram;
    int* prefix_sum;
    int* reordered_result;
    //we'll see if this works directly... if not, switch back to the default memcpy method
    gpuErrchk(cudaMallocHost((void**)&h_histogram, sizeof(int)*numPartitions)  ,"malloc stage");//also use pinned memory
    gpuErrchk(cudaMallocHost((void**)&prefix_sum, sizeof(int)*numPartitions)  ,"malloc stage");//also use pinned memory
    gpuErrchk(cudaMallocHost((void**)&reordered_result, sizeof(int)*rSize)  ,"malloc stage");//also use pinned memory
    

    //begin cuda kernel
    //for now, we use warp size 32
    int blocksize = 32;
    int POOL_SIZE = 128;
    int blockcount;
    //only if we're dealing with really big numbers atm (or whatever threshold we set here), do we concern ourselves with pooling
    if(!FORCE_POOLING && rSize < MAX_THREAD_COUNT)
    {
        printf("disable pooling for array size: %d!\n", rSize);
        POOL_SIZE = -1;
        blockcount = (int)ceil( (double)rSize/ (double) blocksize);
    }
    else
    {
        printf("enable pooling for array size: %d!\n", rSize);
        blockcount = (int) ceil( (double)rSize/ (double) blocksize / (double) POOL_SIZE);
    }
    printf("block count: %d\n", blockcount);
    int numbits =(int)log2((double)numPartitions);
    printf("num bits: %d\n", numbits);


    histogram<<<blockcount, blocksize>>>(POOL_SIZE, r_h, rSize, numbits, h_histogram);

#if ERROR_CHECK
    gpuErrchk( cudaPeekAtLastError() , "histogram1");
    gpuErrchk( cudaDeviceSynchronize(), "histogram2" );
#endif
    //after this I assume the histogram is setup

    //wait a second these launch configs dont make sense... the histogram is ridiculously small
    //sp use numpartitions instead of blockcount... but what to do about blocksize...?
    //so perhaps then, this only has the grid size, and the block is the size of the grid
    //look at notes above the kernel to get a sense as to why I'm setting up the kernel this way
    // prefixScan<<<numPartitions, blocksize>>>(h_histogram, prefix_sum);
    prefixScan<<< 1, numPartitions, sizeof(int)*numPartitions>>>(h_histogram, numPartitions, prefix_sum);
#if ERROR_CHECK
    gpuErrchk( cudaPeekAtLastError() , "prefixScan");
    gpuErrchk( cudaDeviceSynchronize(), "prefixScan" );
#endif
#if PREFIX_DEBUG
    printf("Resulting Prefix Sum array:\n");
    for(int i = 0; i < numPartitions; i++)
    {
        printf("%d, ", prefix_sum[i]);
    }
    
    printf("\n\n");
#endif
    //after this I assume the prefix sum is setup
    Reorder<<<blockcount, blocksize>>>(POOL_SIZE, r_h, rSize, numbits, prefix_sum, reordered_result);
#if ERROR_CHECK
    gpuErrchk( cudaPeekAtLastError() , "Reorder1");
    gpuErrchk( cudaDeviceSynchronize() , "Reorder2");
#endif


    gpuErrchk(cudaEventRecord(stop, 0), "timing-end");
    gpuErrchk(cudaEventSynchronize(stop), "timing-end");
    float elapsedTime;
    gpuErrchk(cudaEventElapsedTime( &elapsedTime, start, stop), "timing-end");
    printf("CUDA EVENT: Running time for GPU version: %0.5f ms\n", elapsedTime);
    gpuErrchk(cudaEventDestroy(start), "timing-end");
    gpuErrchk(cudaEventDestroy(stop), "timing-end");

    //now we should attempt to print out the results...
    //uhh
    //for now, just print out the arrays
#if ARRAY_DEBUG
    printf("Input: \n");
    for(int i = 0; i < rSize; i++)
    {
        printf("%d, ", r_h[i]);
        if(i+1 % 10 == 0)
            printf("\n");
    }
    printf("\n");
    printf("\n");
#endif
    //also we dont really know how to deal with offsets?
    //is that just printing out the histogram sizes?
    //does pointer offset in this case simply mean index of current partition into the reordered array?
    int currentSum = 0;
    for(int i = 0; i < numPartitions; i++)
    {
        int curval = h_histogram[i];
        printf("Partition %d:\n", i+1);
        printf("    Pointer offset(CPU)   :    %d\n", currentSum);
        printf("    Pointer offset(before):    %d\n", prefix_sum[i] - curval);
        printf("    Pointer offset(after) :    %d\n", prefix_sum[i]);
        printf("    Number of Keys: %d\n", curval);
        currentSum += curval;
    }
    printf("\n");


#if ARRAY_DEBUG
    printf("Output: \n");
    for(int i = 0; i < rSize; i++)
    {
        printf("%d, ", reordered_result[i]);
        if(i+1 % 10 == 0)
            printf("\n");
    }
    printf("\n");
#endif

    printf("************* Total Running Time of Kernel = %0.5f sec *************\n", elapsedTime/1000);
    gpuErrchk(cudaFreeHost(r_h), "freeing");
    gpuErrchk(cudaFreeHost(h_histogram), "freeing");
    gpuErrchk(cudaFreeHost(prefix_sum), "freeing");
    gpuErrchk(cudaFreeHost(reordered_result), "freeing");

    return 0;
}