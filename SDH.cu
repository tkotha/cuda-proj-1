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
#define BLOCK_SIZE 128 /*This is temporary until I can a) make sure the basic algorithm is correct and b)I've made sure i know how to dynamically allocate shared memory
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
double   PDH_res;		/* value of w                             */

// atom * atom_list;		/* list of all data points                */
// atom * d_atom_list;

//SOA version of atom_list
double * atom_x_list;
double * atom_y_list;
double * atom_z_list;

//SOA version of d_atom_list
double * d_atom_x_list;
double * d_atom_y_list;
double * d_atom_z_list;


/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	// double x1 = atom_list[ind1].x_pos;
	// double x2 = atom_list[ind2].x_pos;
	// double y1 = atom_list[ind1].y_pos;
	// double y2 = atom_list[ind2].y_pos;
	// double z1 = atom_list[ind1].z_pos;
	// double z2 = atom_list[ind2].z_pos;
	double x1 = atom_x_list[ind1];
	double x2 = atom_x_list[ind2];
	double y1 = atom_y_list[ind1];
	double y2 = atom_y_list[ind2];
	double z1 = atom_z_list[ind1];
	double z2 = atom_z_list[ind2];
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
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
	double dist;
	
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
							double* d_atom_x_list, double* d_atom_y_list, double * d_atom_z_list, 
							long long acnt, double res)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int j, h_pos;
	double dist;
	double x1;
	double x2;
	double y1;
	double y2;
	double z1;
	double z2;
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
							double* d_atom_x_list, double* d_atom_y_list, double * d_atom_z_list, 
							long long acnt, double res)//,
							 //int numBlocks, int blockSize)
{
	extern __shared__ double R[];	
							//the size of this should be 3*BLOCK_SIZE*sizeof(double), to house the three arrays in shared memory	
							//where t is a specific index into the 'atom' array
							//
							//the rth x array should be accessed by R[t + 3*BLOCK_SIZE]				
							//the rth y array should be accessed by R[t + BLOCK_SIZE + 3*BLOCK_SIZE]	
							//the rth z array should be accessed by R[t + BLOCK_SIZE*2 + 3*BLOCK_SIZE]
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, h_pos;
	int i_id, j_id;
	int t = threadIdx.x;
	double  Lx, Ly, Lz, Rx, Ry, Rz;
	double dist;
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
					Rx = R[j];
					Ry = R[j + blockDim.x];
					Rz = R[j + blockDim.x*2];

					dist = sqrt((Lx - Rx)*(Lx-Rx) + (Ly - Ry)*(Ly - Ry) + (Lz - Rz)*(Lz - Rz));

					h_pos = (int)(dist/res);
					atomicAdd(&d_histogram[h_pos], 1);
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
				Rx = R[i];
				Ry = R[i + blockDim.x];
				Rz = R[i + blockDim.x*2];
				dist = sqrt((Lx - Rx)*(Lx-Rx) + (Ly - Ry)*(Ly - Ry) + (Lz - Rz)*(Lz - Rz));

				h_pos = (int)(dist/res);
				atomicAdd(&d_histogram[h_pos], 1);	
			}
		}
	}
}

//now for histogram privitization
//step 1: have it be correct -- apparently it's fine with multiples of block size, but it has very tiny difference with non multiples
			//this smells like an edge case
			//this small error is only introduced when i attempt to do the histogram priv portion
			//doesnt seem to be an off by 1 error in terms of shared mem allocation
//step 2: make sure it is actually faster than tiled
//step 3: make simple optimizations, like reducing actual size of histogram to store multiple copies
//step 4: reduce register count if possible
__global__ void PDH_kernel4(unsigned long long* d_histogram,
							double* d_atom_x_list, double* d_atom_y_list, double* d_atom_z_list,
							long long acnt, double res, int histSize)
{
	extern __shared__ double shmem[];
	double* R = shmem;
	unsigned long long * sh_hist = (unsigned long long *)(R + 3*blockDim.x);

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, h_pos;
	int i_id, j_id;
	int t = threadIdx.x;
	double Lx, Ly, Lz, Rx, Ry, Rz;
	double dist;

	//initialize the shared histogram to 0
	for(i = t; i < histSize; i += blockDim.x)
	{
		sh_hist[i] = 0;
	}
	__syncthreads();

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
					Rx = R[j];
					Ry = R[j + blockDim.x];
					Rz = R[j + blockDim.x*2];

					dist = sqrt((Lx - Rx)*(Lx-Rx) + (Ly - Ry)*(Ly - Ry) + (Lz - Rz)*(Lz - Rz));

					h_pos = (int)(dist/res);
					atomicAdd(&sh_hist[h_pos], 1);
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
				Rx = R[i];
				Ry = R[i + blockDim.x];
				Rz = R[i + blockDim.x*2];
				dist = sqrt((Lx - Rx)*(Lx-Rx) + (Ly - Ry)*(Ly - Ry) + (Lz - Rz)*(Lz - Rz));

				h_pos = (int)(dist/res);
				atomicAdd(&sh_hist[h_pos], 1);	
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
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
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

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram =    (unsigned long long *)malloc(sizeof(unsigned long long)*num_buckets);

	// atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	atom_x_list = (double *)malloc(sizeof(double)*PDH_acnt);
	atom_y_list = (double *)malloc(sizeof(double)*PDH_acnt);
	atom_z_list = (double *)malloc(sizeof(double)*PDH_acnt);


	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_x_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_y_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_z_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */

	printf("Starting CPU...\n");
	gettimeofday(&startTime, &Idunno);
	
	 /*call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram(histogram);
	printf("Starting GPU...\n");

	//cudaDeviceReset();
	//gpu code--------------------------------------------------------------------------------
	h_gpu_histogram = (unsigned long long *)malloc(sizeof(unsigned long long)*num_buckets);

	//copy the atomlist over from host to device
	// cudaMalloc((void**)&d_atom_list, sizeof(atom)*PDH_acnt);
	// cudaMemcpy(d_atom_list, atom_list, sizeof(atom)*PDH_acnt, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_atom_x_list, sizeof(double)*PDH_acnt);
	cudaMemcpy(d_atom_x_list, atom_x_list, sizeof(double)*PDH_acnt, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_atom_y_list, sizeof(double)*PDH_acnt);
	cudaMemcpy(d_atom_y_list, atom_y_list, sizeof(double)*PDH_acnt, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_atom_z_list, sizeof(double)*PDH_acnt);
	cudaMemcpy(d_atom_z_list, atom_z_list, sizeof(double)*PDH_acnt, cudaMemcpyHostToDevice);


	//allocate the histogram data on the device
	cudaMalloc((void**)&d_gpu_histogram, sizeof(unsigned long long)*num_buckets);
	cudaMemset(d_gpu_histogram, 0, sizeof(unsigned long long)*num_buckets);
	cudaMemcpy(d_gpu_histogram, h_gpu_histogram, sizeof(unsigned long long)*num_buckets,cudaMemcpyHostToDevice);

	//start the timer
	gettimeofday(&startTime, &Idunno);



	
	int blockcount = (int)ceil(PDH_acnt / (float) BLOCK_SIZE);
	int shmemsize3 = BLOCK_SIZE*3*sizeof(double);	//this means each 'block' in the shared memory should be about 512 bytes right now, assuming 6400 points
	int shmemsize4 = (BLOCK_SIZE*3)*sizeof(double) + sizeof(unsigned long long)*num_buckets;	//this means each 'block' in the shared memory should be about 512 bytes right now, assuming 6400 points
	printf("blockcount: %d\n",blockcount);
	printf("shmemsize3:  %d\n", shmemsize3);
	printf("shmemsize4:  %d\n", shmemsize4);
	//run the kernel

	// PDH_kernel<<<blockcount, BLOCK_SIZE>>>(d_gpu_histogram, d_atom_x_list, d_atom_y_list, d_atom_z_list, PDH_acnt, PDH_res);
	// PDH_kernel2 <<<blockcount, BLOCK_SIZE, 2*shmemsize>>> //for now, we're allocating blocks for both L and R
	// 	(d_gpu_histogram, 
	// 	 d_atom_x_list, d_atom_y_list, d_atom_z_list, 
	// 	 PDH_acnt, PDH_res,
	// 	 blockcount, BLOCK_SIZE);

	PDH_kernel3 <<<blockcount, BLOCK_SIZE, shmemsize3>>> //now we try and use just R
	(d_gpu_histogram, 
		d_atom_x_list, d_atom_y_list, d_atom_z_list, 
		PDH_acnt, PDH_res);

	// PDH_kernel4 <<<blockcount, BLOCK_SIZE, shmemsize4>>> //now we try to privatize the histogram
	// (d_gpu_histogram, 
	// 	d_atom_x_list, d_atom_y_list, d_atom_z_list, 
	// 	PDH_acnt, PDH_res, num_buckets);


	//copy the histogram results back from gpu over to cpu
	cudaMemcpy(h_gpu_histogram, d_gpu_histogram, sizeof(unsigned long long)*num_buckets, cudaMemcpyDeviceToHost);

	//check total running time
	report_running_time_GPU();

	//print out the resulting histogram from the GPU
	output_histogram(h_gpu_histogram);

	//difference calculation--------------------------------------------------------------------------------
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


