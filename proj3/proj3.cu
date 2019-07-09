#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#include <assert.h>


#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))
#define ARRAY_DEBUG 1
#define PREFIX_DEBUG 0
#define HIST_DEBUG 0
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

//helper function provided by cuda 7.5 shuffle scan sample code
static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
    return ((dividend % divisor) == 0) ?
           (dividend / divisor) :
           (dividend / divisor + 1);
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
        int h = bfe(i_r_h[k], 32, i_numPartitions);    //i assume start value is 0...?
                                                       //nope... it's 32 i think
        atomicAdd(&o_histogram[h], 1);
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

//does this work....? iunno.... but it seems to not crash on input size of 10... which is a start i guess
//howver, upon inspection, it is probably dead wrong....
__global__ void prefixScan(int* i_histogram, int n, int* o_prefix_sum)
{
    //maybe i need multiple device kernels for this to work...
    extern __shared__ int temp[];

    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    // temp[pout*n + thid] = (thid > 0) ? (thid < n) ? i_histogram[thid-1] : 0 : 0;
    temp[pout*n + thid] = (thid > 0) ? i_histogram[thid-1] : 0;
    __syncthreads();

    int offset;
    for(offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout;
        pin = 1 - pout;

        if(thid >= offset)
        {
            temp[pout * n + thid] += temp[pin*n+thid - offset];
        }
        else
        {
            temp[pout * n + thid] = temp[pin*n+thid];
        }
        __syncthreads();
    }

    // if(thid < n)
        o_prefix_sum[thid] = temp[pout*n+thid];
    
}

//define the reorder kernel here
__global__ void Reorder(int* i_r_h, int i_rh_size, int i_numPartitions, int* i_prefix_sum, int* o_r_h)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if(k < i_rh_size)
    {
        int kval = i_r_h[k];
        int h = bfe(kval, 32, i_numPartitions);     //i assume start value is 32 here as well, if we're using the same logic from histogram kernel
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
    int* h_histogram;
    int* prefix_sum;
    int* reordered_result;
    //we'll see if this works directly... if not, switch back to the default memcpy method
    cudaMallocHost((void**)&h_histogram, sizeof(int)*numPartitions);  //also use pinned memory
    cudaMallocHost((void**)&prefix_sum, sizeof(int)*numPartitions);  //also use pinned memory
    cudaMallocHost((void**)&reordered_result, sizeof(int)*rSize);  //also use pinned memory
    

    //begin cuda kernel
    //for now, we use warp size 32
    int blocksize = 32;
    int blockcount = (int)ceil( (double)rSize/ (double) blocksize);


    histogram<<<blockcount, blocksize>>>(r_h, rSize, numPartitions, h_histogram);

    //after this I assume the histogram is setup

    //wait a second these launch configs dont make sense... the histogram is ridiculously small
    //sp use numpartitions instead of blockcount... but what to do about blocksize...?
    //so perhaps then, this only has the grid size, and the block is the size of the grid
    //look at notes above the kernel to get a sense as to why I'm setting up the kernel this way
    // prefixScan<<<numPartitions, blocksize>>>(h_histogram, prefix_sum);
    prefixScan<<< 1, numPartitions, numPartitions>>>(h_histogram, numPartitions, prefix_sum);

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
        printf("    Pointer offset(CPU):    %d\n", currentSum);
        printf("    Pointer offset(GPU):    %d\n", prefix_sum[i]);
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
    cudaFreeHost(r_h);
    cudaFreeHost(h_histogram);
    cudaFreeHost(prefix_sum);
    cudaFreeHost(reordered_result);

    return 0;
}