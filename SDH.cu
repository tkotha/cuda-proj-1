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
#define BLOCK_COUNT 256 /*This is temporary until I can a) make sure the basic algorithm is correct and b)I've made sure i know how to dynamically allocate shared memory
/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
bucket * h_gpu_histogram;
bucket * d_gpu_histogram;
bucket * diff_histogram;

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
			histogram[h_pos].d_cnt++;
		} 
		
	}
	return 0;
}

/*
	SDH kernel - a really crappy one
*/

__global__ void PDH_kernel(bucket* d_histogram, double* d_atom_x_list, double* d_atom_y_list, double * d_atom_z_list, long long acnt, double res)
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
			// x1 = d_atom_list[id].x_pos;
			// x2 = d_atom_list[j].x_pos;
			// y1 = d_atom_list[id].y_pos;
			// y2 = d_atom_list[j].y_pos;
			// z1 = d_atom_list[id].z_pos;
			// z2 = d_atom_list[j].z_pos;
			x1 = d_atom_x_list[id];
			x2 = d_atom_x_list[j];
			y1 = d_atom_y_list[id];
			y2 = d_atom_y_list[j];
			z1 = d_atom_z_list[id];
			z2 = d_atom_z_list[j];
			dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
			h_pos = (int) (dist / res);
			atomicAdd((unsigned long long int*)&d_histogram[h_pos].d_cnt,1);
		}
}

/*
	An attempt at an improved version of the kernel
	step 1: get block tiling to work
	step 2: get histogram privitization to work
*/

__global__ void PDH_kernel2(bucket* d_histogram, double* d_atom_x_list, double* d_atom_y_list, double * d_atom_z_list, long long acnt, double res, int blockcount)
{
	//our location in the global atom list
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	//load our single data point
	double x1 = d_atom_x_list[id];
	double y1 = d_atom_y_list[id];
	double z1 = d_atom_z_list[id];
	double x2,y2,z2;
	int i, j, h_pos;
	double dist;

	//our single block tile for handling the nested loop... wait how do we do this at runtime?
	// __shared__ double r_block_x[blockDim.x];
	// __shared__ double r_block_y[blockDim.x];
	// __shared__ double r_block_z[blockDim.x];

	//here's the plan, we're just going to use this guy directly!
	//all we have to do is 'access' the correct portions of shared mem
	//so, threadid + blockdim*<axes>
	//where axes is 0=x, 1 = y, and z = 2
	//...
	//ok, for right now, just to see if the basic code even works, we'll keep to a static block size
	//say 256
	//once we're sure this much is correct, we'll work out making it dynamically sizeable
	 __shared__ double xblock[BLOCK_COUNT];
	 __shared__ double yblock[BLOCK_COUNT];
	 __shared__ double zblock[BLOCK_COUNT];
	 // double *xblock = r_block;
	 // double *yblock = (double*)&xblock[BLOCK_COUNT];
	 // double *zblock = (double*)&yblock[BLOCK_COUNT];

	 //small debug logic
	 if(threadIdx.x == 0 && blockIdx.x == 0)
	 {
		printf("BLOCK COUNT: %d\n", BLOCK_COUNT);
		printf("GRID SIZE: %d\n", gridDim.x);
	 }
	
	//interblock for loop, for the M value, use the grid's dimensions
	for(i = blockIdx.x+1; i < gridDim.x; i++)
	{

		xblock[threadIdx.x] = 	d_atom_x_list[blockDim.x*i + threadIdx.x];
		yblock[threadIdx.x] = 	d_atom_y_list[blockDim.x*i + threadIdx.x];
		zblock[threadIdx.x] = 	d_atom_z_list[blockDim.x*i + threadIdx.x];

		__syncthreads();
		for(j = 0; j < blockDim.x; j++)
		{
			//this func
			x2 = xblock[j];
			y2 = yblock[j];
			z2 = zblock[j];
			dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
			if(threadIdx.x == 0)
			{
				printf("dist: %d\n", dist);
			}
			//atomic add
			h_pos = (int)(dist/res);
			atomicAdd((unsigned long long int*)&d_histogram[h_pos].d_cnt,1);
		}
	}

	//intrablock for loop
	xblock[threadIdx.x] = 	d_atom_x_list[id];
	yblock[threadIdx.x] = 	d_atom_y_list[id];
	zblock[threadIdx.x] = 	d_atom_z_list[id];

	__syncthreads();
	for(i = threadIdx.x +1; i < blockDim.x; i++)
	{
		//this func
		x2 = xblock[j];
		y2 = yblock[j];
		z2 = zblock[j];
		dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
		//atomic add
		h_pos = (int)(dist/res);
		atomicAdd((unsigned long long int*)&d_histogram[h_pos].d_cnt,1);
	}


	//write output back to histogram... not yet! we havent gotten to the privatized histogram yet!
	//__syncthreads();
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
void output_histogram(bucket* histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
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
	histogram =       (bucket *)malloc(sizeof(bucket)*num_buckets);

	// atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	atom_x_list = (double *)malloc(sizeof(double)*PDH_acnt);
	atom_y_list = (double *)malloc(sizeof(double)*PDH_acnt);
	atom_z_list = (double *)malloc(sizeof(double)*PDH_acnt);


	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		// atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		// atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		// atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;

		atom_x_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_y_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_z_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	printf("Starting CPU...\n");
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram(histogram);
	printf("Starting GPU...\n");

	//cudaDeviceReset();
	//gpu code--------------------------------------------------------------------------------
	h_gpu_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

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
	cudaMalloc((void**)&d_gpu_histogram, sizeof(bucket)*num_buckets);
	cudaMemcpy(d_gpu_histogram, h_gpu_histogram, sizeof(bucket)*num_buckets,cudaMemcpyHostToDevice);

	//start the timer
	gettimeofday(&startTime, &Idunno);
	int blockcount = BLOCK_COUNT;

	//run the kernel
	// PDH_kernel<<<ceil(PDH_acnt/256.0), 256>>>(d_gpu_histogram, d_atom_list, PDH_acnt, PDH_res);
	// PDH_kernel<<<ceil(PDH_acnt/((float)blockcount)), blockcount>>>(d_gpu_histogram, d_atom_x_list, d_atom_y_list, d_atom_z_list, PDH_acnt, PDH_res);
	PDH_kernel2<<<ceil(PDH_acnt/(((float)blockcount))), blockcount>>>
	(d_gpu_histogram, d_atom_x_list, d_atom_y_list, d_atom_z_list, PDH_acnt, PDH_res, blockcount);

	//copy the histogram results back from gpu over to cpu
	cudaMemcpy(h_gpu_histogram, d_gpu_histogram, sizeof(bucket)*num_buckets, cudaMemcpyDeviceToHost);

	//check total running time
	report_running_time_GPU();

	//print out the resulting histogram from the GPU
	output_histogram(h_gpu_histogram);

	//difference calculation--------------------------------------------------------------------------------
	printf("Difference: \n");
	diff_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	int bi;
	for(bi = 0; bi < num_buckets; bi++)
	{
		diff_histogram[bi].d_cnt = histogram[bi].d_cnt - h_gpu_histogram[bi].d_cnt;
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

	return 0;
}


