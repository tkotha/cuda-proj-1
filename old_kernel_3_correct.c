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
