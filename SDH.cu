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
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

//add vars for gpu
bucket * h_gpu_histogram;
bucket * d_gpu_histogram;
atom * d_atom_list;			//we will use the atom list as the original code defines it

//add another histogram to get the differences
bucket * h_diff_histogram;

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


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
	global kernel for calculating the SDH
	it just occurred to me that this makes no effort to protect against conflicting writes in the histogram
	... oh well, we'll see what happens
*/
__global__ void PDH_GPU(bucket * d_histogram, atom * d_atom_list, long long acnt, double res){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int j, h_pos;
	double dist;

	double x1 ;
	double x2 ;
	double y1 ;
	double y2 ;
	double z1 ;
	double z2 ;
	if(id >= acnt) return;
	for(j = id+1; j < acnt; j++)
	{
		
		x1 = d_atom_list[id].x_pos;
		x2 = d_atom_list[j].x_pos;
		y1 = d_atom_list[id].y_pos;
		y2 = d_atom_list[j].y_pos;
		z1 = d_atom_list[id].z_pos;
		z2 = d_atom_list[j].z_pos;

		dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));	//does this require a float parameter, or should double be fine?
		h_pos = (int) (dist / res);
		//histogram[h_pos].d_cnt++;
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
void output_histogram(bucket * hist, int useTValue){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", hist[i].d_cnt);
		total_cnt += hist[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
		{
			
			printf("\n T:%lld \n", total_cnt);

		}
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	int atomsize = sizeof(atom)*PDH_acnt;
	int bucketsize = sizeof(bucket)*num_buckets;

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(bucketsize);

	atom_list = (atom *)malloc(atomsize);

	//allocate any needed host side gpu vars here
	h_gpu_histogram = (bucket *)malloc(bucketsize);
	h_diff_histogram = (bucket *)malloc(bucketsize);
	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram(histogram, 1);


	//now for the gpu part
	//allocate relevant information
	
	cudaMalloc((void**) &d_gpu_histogram, bucketsize);
	cudaMalloc((void**) &d_atom_list, atomsize);

	//copy host atomlist over to the GPU
	cudaMemcpy(d_atom_list, atom_list, atomsize, cudaMemcpyHostToDevice);

	//start the timer
	gettimeofday(&startTime, &Idunno);

	//call the kernel here
	//i have no idea if this is optimal, I'm just throwing in a number
	//the threads are split up based on the # of atoms, as each atom needs to process its collision pairs and update the histogram
	//PDH_GPU<<<ceil(PDH_acnt/256.0), 256>>>(d_gpu_histogram, d_atom_list, PDH_acnt,PDH_res);

	//copy the results from the GPU back
	//cudaMemcpy(h_gpu_histogram, d_gpu_histogram, bucketsize, cudaMemcpyDeviceToHost);
	//check the total running time
	report_running_time_GPU();

	//dont forget to free your crap from the GPU
	cudaFree(d_gpu_histogram);
	cudaFree(d_atom_list);

	//print out the histogram
	output_histogram(h_gpu_histogram, 1);



	//calculate the differences between the two and store it
	int hi;
	for(hi = 0; hi < num_buckets; hi ++)
	{
		h_diff_histogram[hi].d_cnt = histogram[hi].d_cnt - h_gpu_histogram[hi].d_cnt;
	}


	//now for the comparisons between the CPU and GPU
	printf("Now the differences between the two histograms: \n");
	output_histogram(h_diff_histogram, 0);
	
	return 0;
}


