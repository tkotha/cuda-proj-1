/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */
#define COMPARE_CPU 1
#define KERNELTYPE 4

#define ATOM_DIM double
#define ATOM_ZERO 0.0
#define SQRT_CPU sqrt
#define SQRT sqrt

// #define ATOM_DIM float
// #define ATOM_ZERO 0.0
// #define SQRT_CPU sqrtf //yeaaa dont even think about using this, it's a red herring
// #define SQRT_CPU sqrt
// #define SQRT __fsqrt_rn


/* descriptors for single atom in the tree */
// typedef struct atomdesc {
// 	double x_pos;
// 	double y_pos;
// 	double z_pos;
// } atom;

// typedef struct hist_entry{
// 	//float min;
// 	//float max;
// 	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
// } bucket;


// bucket * histogram;		/* list of all buckets in the histogram   */
// bucket * h_gpu_histogram;
// bucket * d_gpu_histogram;
// bucket * diff_histogram;
// void checkCudaError(cudaError_t e, char in[]) {
// 	if (e != cudaSuccess) {
// 		printf("CUDA Error: %s, %s \n", in, cudaGetErrorString(e));
// 		exit(EXIT_FAILURE);
// 	}
// }

unsigned long long * histogram;
unsigned long long * h_gpu_histogram;
unsigned long long * d_gpu_histogram;
unsigned long long * diff_histogram;

long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
ATOM_DIM   PDH_res;		/* value of w                             */
int BLOCK_SIZE;
// atom * atom_list;		/* list of all data points                */
// atom * d_atom_list;

//SOA version of atom_list
ATOM_DIM * atom_x_list;
ATOM_DIM * atom_y_list;
ATOM_DIM * atom_z_list;

//SOA version of d_atom_list
ATOM_DIM * d_atom_x_list;
ATOM_DIM * d_atom_y_list;
ATOM_DIM * d_atom_z_list;


/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
ATOM_DIM p2p_distance(int ind1, int ind2) {
	
	// double x1 = atom_list[ind1].x_pos;
	// double x2 = atom_list[ind2].x_pos;
	// double y1 = atom_list[ind1].y_pos;
	// double y2 = atom_list[ind2].y_pos;
	// double z1 = atom_list[ind1].z_pos;
	// double z2 = atom_list[ind2].z_pos;
	ATOM_DIM x1 = atom_x_list[ind1];
	ATOM_DIM x2 = atom_x_list[ind2];
	ATOM_DIM y1 = atom_y_list[ind1];
	ATOM_DIM y2 = atom_y_list[ind2];
	ATOM_DIM z1 = atom_z_list[ind1];
	ATOM_DIM z2 = atom_z_list[ind2];
	return SQRT_CPU((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

//get cuda error
// void cuda_err(cudaError_t e, char in[])
// {
// 	if (e != cudaSuccess){
// 		printf("CUDA Error: %s, %s \n", in, cudaGetErrorString(e));
// 		exit(EXIT_FAILURE);
// 	}
// }


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	ATOM_DIM dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			// histogram[h_pos].d_cnt++;
			histogram[h_pos]++;
		} 
		
	}
	return 0;
}



/*
	SDH kernel - a really crappy one
*/

__global__ void PDH_kernel(unsigned long long* d_histogram, 
							ATOM_DIM* d_atom_x_list, ATOM_DIM* d_atom_y_list, ATOM_DIM * d_atom_z_list, 
							long long acnt, ATOM_DIM res)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int j, h_pos;
	ATOM_DIM dist;
	ATOM_DIM x1;
	ATOM_DIM x2;
	ATOM_DIM y1;
	ATOM_DIM y2;
	ATOM_DIM z1;
	ATOM_DIM z2;
	if(id < acnt) 
		for(j = id+1; j < acnt; j++)
		{
			x1 = d_atom_x_list[id];
			x2 = d_atom_x_list[j];
			y1 = d_atom_y_list[id];
			y2 = d_atom_y_list[j];
			z1 = d_atom_z_list[id];
			z2 = d_atom_z_list[j];
			dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
			h_pos = (int) (dist / res);
			// atomicAdd((unsigned long long int*)&d_histogram[h_pos].d_cnt,1);
			atomicAdd(&d_histogram[h_pos], 1);
		}
}

/*
	An attempt at an improved version of the kernel
	step 1: get block tiling to work
	step 2: get histogram privitization to work
*/

//note: when we go for correctness, this is the kernel we will work from. This way we have the best chance of finding the edge case easier
//ok, so syncing threads made the 6400 point case correct, but we are still dead wrong with the 10000 point case for some reason
//update: now the kernel is correct, and we match junyi's performance
__global__ void PDH_kernel3(unsigned long long* d_histogram, 
							ATOM_DIM* d_atom_x_list, ATOM_DIM* d_atom_y_list, ATOM_DIM * d_atom_z_list, 
							long long acnt, ATOM_DIM res)//,
							 //int numBlocks, int blockSize)
{
	extern __shared__ ATOM_DIM R[];	
							//the size of this should be 3*BLOCK_SIZE*sizeof(double), to house the three arrays in shared memory	
							//where t is a specific index into the 'atom' array
							//
							//the rth x array should be accessed by R[t + 3*BLOCK_SIZE]				
							//the rth y array should be accessed by R[t + BLOCK_SIZE + 3*BLOCK_SIZE]	
							//the rth z array should be accessed by R[t + BLOCK_SIZE*2 + 3*BLOCK_SIZE]
	int cur_id = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, h_pos;
	//int i_id, j_id;
	// int cur_id;
	ATOM_DIM  Lx, Ly, Lz, Rt;//, Rx, Ry, Rz;
	ATOM_DIM dist;
	if(cur_id < acnt)
	{
		Lx = d_atom_x_list[cur_id];
		Ly = d_atom_y_list[cur_id];
		Lz = d_atom_z_list[cur_id];
		for(i = blockIdx.x +1; i < gridDim.x; i++)
		{
			cur_id = i * blockDim.x + threadIdx.x;	//only valid threads may load into shared memory for block i
			if(cur_id < acnt)					
			{
				R[threadIdx.x] 				= d_atom_x_list[cur_id];
				R[threadIdx.x + blockDim.x]	= d_atom_y_list[cur_id];
				R[threadIdx.x + blockDim.x*2]	= d_atom_z_list[cur_id];
			}
			__syncthreads();
			for(j = 0; j < blockDim.x; j++) 
			{
				cur_id = i * blockDim.x + j;	//now this prevents us from writing junk data for thread j
				if(cur_id < acnt)
				{
					// Rx = R[j];
					// Ry = R[j + blockDim.x];
					// Rz = R[j + blockDim.x*2];
					// dist = sqrt((Lx - Rx)*(Lx-Rx) + (Ly - Ry)*(Ly - Ry) + (Lz - Rz)*(Lz - Rz));
					dist = ATOM_ZERO;
					//dist = 0f;
					//Rx
					Rt = Lx - R[j];
					Rt *= Rt;
					dist += Rt;

					//Ry
					Rt = Ly - R[j + blockDim.x];
					Rt *= Rt;
					dist += Rt;

					//Rz
					Rt = Lz - R[j + blockDim.x*2];
					Rt *= Rt;
					dist += Rt;

					dist = SQRT(dist);

					h_pos = (int)(dist/res);
					atomicAdd(&d_histogram[h_pos], 1);
				}
			}
			__syncthreads();
			
		}

		//now load the L values into R
		R[threadIdx.x] = Lx;
		R[threadIdx.x + blockDim.x] = Ly;
		R[threadIdx.x + blockDim.x*2] = Lz;
		__syncthreads();
		for(i = threadIdx.x+ 1; i < blockDim.x; i++)
		{
			cur_id = blockIdx.x * blockDim.x + i;	//we only proceed with valid threads for each thread i
			if(cur_id < acnt)
			{
				// Rx = R[i];
				// Ry = R[i + blockDim.x];
				// Rz = R[i + blockDim.x*2];
				// dist = sqrt((Lx - Rx)*(Lx-Rx) + (Ly - Ry)*(Ly - Ry) + (Lz - Rz)*(Lz - Rz));
				dist = ATOM_ZERO;
				//Rx
				Rt = Lx - R[i];
				Rt *= Rt;
				dist += Rt;

				//Ry
				Rt = Ly - R[i + blockDim.x];
				Rt *= Rt;
				dist += Rt;

				//Rz
				Rt = Lz - R[i + blockDim.x*2];
				Rt *= Rt;
				dist += Rt;

				dist = SQRT(dist);

				h_pos = (int)(dist/res);
				atomicAdd(&d_histogram[h_pos], 1);	
			}
		}
	}
}

//now for histogram privitization
//step 1: have it be correct -- apparently it's fine with multiples of block size, but it has very tiny difference with non multiples
			//this smells like an edge case
			//this small error is only introduced when i attempt to do the histogram priv portion. it doesnt exist if i go back to writing in the global histogram
			//doesnt seem to be an off by 1 error in terms of shared mem allocation
// till i have a better handle on what's going on here, I may need to fix the block size and numbuckets size
			//interesting... this kernel is in fact correct at blocksizes of 32 and 512. Not sure why but I'll go with it
			//incidently, it's also its fastest at 32... good enough for me
			//however, its still slower than my tiled kernel at the same blocksize(by .5 ms)

// so that I can debug the actual shared memory portion
//step 2: make sure it is actually faster than tiled
			//presently it is not

//step 3: make simple optimizations, like reducing actual size of histogram to store multiple copies
//step 4: reduce register count if possible
//unsigned long long is typically 8 bytes. int is typically 4 bytes, short is 2 bytes, and char is 1 byte
// double is 8 bytes, float is 4 bytes

//here we make a very strong assumption that the kernel block size is 32 only!

//this seems to behave correctly if the blocksize is 512 or 32


__global__ void PDH_kernel4(unsigned long long* d_histogram,
							ATOM_DIM* d_atom_x_list, ATOM_DIM* d_atom_y_list, ATOM_DIM* d_atom_z_list,
							long long acnt, ATOM_DIM res, int histSize)
{
	extern __shared__ ATOM_DIM shmem[];
	//for now assume a block count of 157 and 80 (based on 10000 pts, 500.0 resolution, and 64 blocks)
	// __shared__ int* shmem[(157*3)*sizeof(double) + sizeof(/*unsigned long long*/ int)*80];
	// double* R = (double*)shmem;
	ATOM_DIM* R = shmem;
	//2 copies of histogram, but we use one pointer
	int * sh_hist = (int *)(R + 3*blockDim.x);


	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, h_pos;
	int i_id, j_id;
	int t = threadIdx.x;
	ATOM_DIM Lx, Ly, Lz, Rx, Ry, Rz;
	ATOM_DIM dist;

	//initialize the shared histogram to 0
	for(i = t; i < histSize; i += blockDim.x)
	{
		sh_hist[i] = 0;
	}
	//do tiled algorithm with sh_hist
	if(id < acnt)
	{
		Lx = d_atom_x_list[id];
		Ly = d_atom_y_list[id];
		Lz = d_atom_z_list[id];
		for(i = blockIdx.x +1; i < gridDim.x; i++)
		{
			i_id = i * blockDim.x + t;	//only valid threads may load into shared memory
			if(i_id < acnt)					
			{
				R[t] 				= d_atom_x_list[i_id];
				R[t + blockDim.x]	= d_atom_y_list[i_id];
				R[t + blockDim.x*2]	= d_atom_z_list[i_id];
			}
			__syncthreads();
			for(j = 0; j < blockDim.x; j++) 
			{
				j_id = i * blockDim.x + j;	//now this prevents us from writing junk data
				if(j_id < acnt)
				{
					/* DISTANCE FUNCTION */
					Rx = R[j];
					Rx = Lx - Rx;
					Rx *= Rx;

					Ry = R[j + blockDim.x];
					Ry = Ly - Ry;
					Ry *= Ry;

					Rz = R[j + blockDim.x*2];
					Rz = Lz - Rz;
					Rz *= Rz;

					dist = SQRT( ((Rx) + (Ry) + (Rz)) );
					h_pos = (int)(dist/res);
					/* END DISTANCE FUNCTION */

					
					atomicAdd((int*)&sh_hist[h_pos], 1);
					// atomicAdd(&d_histogram[h_pos], 1);
				}
			}
			__syncthreads();
			
		}

		//now load the L values into R
		R[t] = Lx;
		R[t + blockDim.x] = Ly;
		R[t + blockDim.x*2] = Lz;
		__syncthreads();
		for(i = t+ 1; i < blockDim.x; i++)
		{
			i_id = blockIdx.x * blockDim.x + i;
			if(i_id < acnt)
			{

				/* DISTANCE FUNCTION */
				Rx = R[i];
				Rx = Lx - Rx;
				Rx *= Rx;


				Ry = R[i + blockDim.x];
				Ry = Ly - Ry;
				Ry *= Ry;
				
				Rz = R[i + blockDim.x*2];
				Rz = Lz - Rz;
				Rz *= Rz;
				
				dist = SQRT( ((Rx) + (Ry) + (Rz)) );
				/* END DISTANCE FUNCTION */

				h_pos = (int)(dist/res);
				atomicAdd((int*)&sh_hist[h_pos], 1);
				// atomicAdd(&d_histogram[h_pos], 1);
			}
			
		}
	}



	//now write back to output
	__syncthreads();
	for(i = t; i < histSize; i += blockDim.x)
	{
		atomicAdd(&d_histogram[i], sh_hist[i]);
	}

}



/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld seconds\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

double report_running_time_GPU() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(unsigned long long* histogram){
	int i; 
	unsigned long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15llu ", histogram[i]);//histogram[i].d_cnt);
		total_cnt += histogram[i];//histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%llu \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;
	// #define BLOCK_SIZE 64 /*This is temporary until I can a) make sure the basic algorithm is correct and b)I've made sure i know how to dynamically allocate shared memory
	if(argc != 4)
	{
		printf("ERROR: Must specify 3 arguments after ./SDH! parameters allowed are (in order):\n sample_num(int), bucket_width(double), block_size(int)\n");
		printf("Example Execution: ./SDH 10000 500.0 64\n This will compute 10000 points,\n using histogram with bucket resolution 500.0,\n using 64 blocks in the GPU\n");
		return 1;
	}
	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	BLOCK_SIZE = atoi(argv[3]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / (double)PDH_res) + 1;
	histogram =    (unsigned long long *)malloc(sizeof(unsigned long long)*num_buckets);

	// atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	atom_x_list = (ATOM_DIM *)malloc(sizeof(ATOM_DIM)*PDH_acnt);
	atom_y_list = (ATOM_DIM *)malloc(sizeof(ATOM_DIM)*PDH_acnt);
	atom_z_list = (ATOM_DIM *)malloc(sizeof(ATOM_DIM)*PDH_acnt);


	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_x_list[i] = (ATOM_DIM)(((ATOM_DIM)(rand()) / RAND_MAX) * BOX_SIZE);
		atom_y_list[i] = (ATOM_DIM)(((ATOM_DIM)(rand()) / RAND_MAX) * BOX_SIZE);
		atom_z_list[i] = (ATOM_DIM)(((ATOM_DIM)(rand()) / RAND_MAX) * BOX_SIZE);
	}
	/* start counting time */

#if COMPARE_CPU
	printf("Starting CPU...\n");
	gettimeofday(&startTime, &Idunno);
	
	 /*call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram(histogram);
#endif


	printf("Starting GPU...\n");

	//cudaDeviceReset();
	//gpu code--------------------------------------------------------------------------------
	h_gpu_histogram = (unsigned long long *)malloc(sizeof(unsigned long long)*num_buckets);

	//copy the atomlist over from host to device
	// cudaMalloc((void**)&d_atom_list, sizeof(atom)*PDH_acnt);
	// cudaMemcpy(d_atom_list, atom_list, sizeof(atom)*PDH_acnt, cudaMemcpyHostToDevice);

	//start the timer
	//gettimeofday(&startTime, &Idunno);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&d_atom_x_list, sizeof(ATOM_DIM)*PDH_acnt);
	cudaMemcpy(d_atom_x_list, atom_x_list, sizeof(ATOM_DIM)*PDH_acnt, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_atom_y_list, sizeof(ATOM_DIM)*PDH_acnt);
	cudaMemcpy(d_atom_y_list, atom_y_list, sizeof(ATOM_DIM)*PDH_acnt, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_atom_z_list, sizeof(ATOM_DIM)*PDH_acnt);
	cudaMemcpy(d_atom_z_list, atom_z_list, sizeof(ATOM_DIM)*PDH_acnt, cudaMemcpyHostToDevice);


	//allocate the histogram data on the device
	cudaMalloc((void**)&d_gpu_histogram, sizeof(unsigned long long)*num_buckets);
	cudaMemset(d_gpu_histogram, 0, sizeof(unsigned long long)*num_buckets);
	cudaMemcpy(d_gpu_histogram, h_gpu_histogram, sizeof(unsigned long long)*num_buckets,cudaMemcpyHostToDevice);


	//Q:i should ask if the cudamalloc, memset, and memcpy should be included in time recording, or if we should do without it
	
	int blockcount = (int)ceil(PDH_acnt / (double) BLOCK_SIZE);
	int shmemsize3 = BLOCK_SIZE*3*sizeof(ATOM_DIM);	//this means each 'block' in the shared memory should be about 512 bytes right now, assuming 6400 points
	int shmemsize4 = (BLOCK_SIZE*3)*sizeof(ATOM_DIM) + sizeof(/*unsigned long long*/ int)*num_buckets;	//this means each 'block' in the shared memory should be about 512 bytes right now, assuming 6400 points
	printf("blockcount: %d\n",blockcount);
	printf("numbuckets: %d\n", num_buckets);
	printf("shmemsize3:  %d\n", shmemsize3);
	printf("shmemsize4:  %d\n", shmemsize4);


	//depending on how this goes, what I may opt for is 'strategizing' my kernels.
	//I will prioritize on getting the current kernel 4 on being as fast as possible (assuming it's correct at blocksize 32)
	//if it can beat kernel 3, then I will conditoinally check for requested blocksize.
	//if it equals 32 then I use kernel 4, otherwise I use kernel 3


	//run the kernel
#if KERNELTYPE == 1
	PDH_kernel<<<blockcount, BLOCK_SIZE>>>(d_gpu_histogram, d_atom_x_list, d_atom_y_list, d_atom_z_list, PDH_acnt, PDH_res);

#elif KERNELTYPE == 3
	PDH_kernel3 <<<blockcount, BLOCK_SIZE, shmemsize3>>> //now we try and use just R
	(d_gpu_histogram, 
		d_atom_x_list, d_atom_y_list, d_atom_z_list, 
		PDH_acnt, PDH_res);

	/*
	for inputsize 10000
		current best timings (of the accurate running configurations):
		1) blocksize 32 : 32.50355 ms		--since this is my best configuration, perhaps I should do a warp specific kernel
		2) blocksize 64 : 33.21270 ms

	for inputsize 512000
		1) blocksize 32 : 66.485 seconds
	*/

#elif KERNELTYPE == 4
	// PDH_kernel4 <<<blockcount, BLOCK_SIZE/*, shmemsize4*/>>> //now we try to privatize the histogram
	//if(BLOCK_SIZE == 32)
		PDH_kernel4 <<<blockcount, BLOCK_SIZE, shmemsize4>>> //now we try to privatize the histogram
		(d_gpu_histogram, 
			d_atom_x_list, d_atom_y_list, d_atom_z_list, 
			PDH_acnt, PDH_res, num_buckets);
	// else
	// {
	// 	printf("ERROR: kernel 4 is only accurate at block size 32! QUIT\n");
	// 	goto cudaFinish;
	// }

	/*
	for inputsize 10000
		current best timings (of the accurate running configurations):
		1) blocksize 32  : 33.01818 ms
		2) blocksize 512 : 42.04790 ms

	for inputsize 512000
		1) blocksize 32 : 68.002 seconds

	*/

#endif

	//copy the histogram results back from gpu over to cpu
	cudaMemcpy(h_gpu_histogram, d_gpu_histogram, sizeof(unsigned long long)*num_buckets, cudaMemcpyDeviceToHost);

	//check total running time
	//report_running_time_GPU();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop);
	printf("CUDA EVENT: Running time for GPU version: %0.5f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//print out the resulting histogram from the GPU
	output_histogram(h_gpu_histogram);

	//difference calculation--------------------------------------------------------------------------------
	
#if COMPARE_CPU
	printf("Difference: \n");
	diff_histogram = (unsigned long long *)malloc(sizeof(unsigned long long)*num_buckets);
	int bi;
	for(bi = 0; bi < num_buckets; bi++)
	{
		// diff_histogram[bi].d_cnt = histogram[bi].d_cnt - h_gpu_histogram[bi].d_cnt;
		if(histogram[bi] > h_gpu_histogram[bi])
			diff_histogram[bi] = histogram[bi] - h_gpu_histogram[bi];
		else
			diff_histogram[bi] = h_gpu_histogram[bi] - histogram[bi];
	}


	output_histogram(diff_histogram);

#endif

	printf("************* Total Running Time of Kernel = %0.5f ms *************\n", elapsedTime);
#if KERNELTYPE == 4
// cudaFinish:
#endif
	cudaFree(d_gpu_histogram);
	cudaFree(d_atom_x_list);
	cudaFree(d_atom_y_list);
	cudaFree(d_atom_z_list);
	free(histogram);
	free(atom_x_list);
	free(atom_y_list);
	free(atom_z_list);
	free(h_gpu_histogram);
	free(diff_histogram); 

	cudaDeviceReset();

	return 0;
}


