#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#include <assert.h>


#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))
#define ARRAY_DEBUG 1
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
__global__ void histogram(int* i_r_h, int i_rh_size, int i_numPartitions ,int* o_histogram)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if(k < i_rh_size)
    {
        int h = bfe(i_r_h[k], 0, i_numPartitions);    //i assume start value is 0...?
        atomicAdd(&o_histogram[h], 1);
    }
}

//define the prefix scan kernel here
//implement it yourself or borrow the code from CUDA samples
__global__ void prefixScan(int* i_histogram, int* o_prefix_sum)
{

}

//define the reorder kernel here
__global__ void Reorder(int* i_r_h, int i_rh_size, int i_numPartitions, int* i_prefix_sum, int* o_r_h)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if(k < i_rh_size)
    {
        int kval = i_r_h[k];
        int h = bfe(kval, 0, i_numPartitions);
        int offset = atomicAdd(&i_prefix_sum[h], 1);
        o_r_h[offset] = kval;
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

    cudaMallocHost((void**)&r_h, sizeof(int)*rSize); //use pinned memory 
    
    dataGenerator(r_h, rSize, 0, 1);
    
    /* your code */
    //i will start measuring time from my code specifically
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    //allocate histogram, prefix sum, and reordered buffer
    int* histogram;
    int* prefix_sum;
    int* reordered_result;
    //we'll see if this works directly... if not, switch back to the default memcpy method
    cudaMallocHost((void**)&histogram, sizeof(int)*numPartitions);  //also use pinned memory
    cudaMallocHost((void**)&prefix_sum, sizeof(int)*numPartitions);  //also use pinned memory
    cudaMallocHost((void**)&reordered_result, sizeof(int)*rSize);  //also use pinned memory
    

    //begin cuda kernel
    //for now, we use warp size 32
    int blocksize = 32;
    int blockcount = (int)ceil( rSize/ (double) blocksize);


    // histogram<<<blockcount, blocksize>>>(r_h, rSize, histogram, numPartitions);

    //after this I assume the histogram is setup

    //wait a second these launch configs dont make sense... the histogram is ridiculously small
    //sp use numpartitions instead of blockcount... but what to do about blocksize...?
    prefixScan<<<numPartitions, blocksize>>>(histogram, prefix_sum);

    //after this I assume the prefix sum is setup

    Reorder<<<blockcount, blocksize>>>(r_h, rSize, numPartitions, prefix_sum, reordered_result);



    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop);
    printf("CUDA EVENT: Running time for GPU version: %0.5f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //now we should attempt to print out the results...
    //uhh
    //for now, just print out the arrays
#if ARRAY_DEBUG
    printf("Input: \n");
    for(int i = 0; i < rSize; i++)
    {
        printf("%d, ", r_h[i]);
        if(i % 10 == 0)
            printf("\n");
    }
#endif
    //also we dont really know how to deal with offsets?
    //is that just printing out the histogram sizes?
    //does pointer offset in this case simply mean index of current partition into the reordered array?
    int currentSum = 0;
    for(int i = 0; i < numPartitions; i++)
    {
        int curval = histogram[i];
        printf("Partition %d:\n", i+1);
        printf("    Pointer offset:    %d\n", currentSum);
        printf("    Number of Keys: %d\n", curval);
        currentSum += curval;
    }
    printf("\n");


#if ARRAY_DEBUG
    printf("Output: \n");
    for(int i = 0; i < rSize; i++)
    {
        printf("%d, ", reordered_result[i]);
        if(i % 10 == 0)
            printf("\n");
    }
#endif

    printf("************* Total Running Time of Kernel = %0.5f sec *************\n", elapsedTime/1000);
    cudaFreeHost(r_h);
    cudaFreeHost(histogram);
    cudaFreeHost(prefix_sum);
    cudaFreeHost(reordered_result);

    return 0;
}