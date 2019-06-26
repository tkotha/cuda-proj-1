__global__ void PDH_kernel4(unsigned long long* d_histogram,
							double* d_atom_x_list, double* d_atom_y_list, double* d_atom_z_list,
							long long acnt, double res, int histSize)
{
	extern __shared__ double shmem[];
	//for now assume a block count of 157 and 80 (based on 10000 pts, 500.0 resolution, and 64 blocks)
	// __shared__ int* shmem[(157*3)*sizeof(double) + sizeof(/*unsigned long long*/ int)*80];
	// double* R = (double*)shmem;
	double* R = shmem;
	//2 copies of histogram, but we use one pointer
	int * sh_hist = (int *)(R + 3*blockDim.x);


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
					Ry = R[j + blockDim.x];
					Rz = R[j + blockDim.x*2];

					dist = sqrt((Lx - Rx)*(Lx-Rx) + (Ly - Ry)*(Ly - Ry) + (Lz - Rz)*(Lz - Rz));
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
				Ry = R[i + blockDim.x];
				Rz = R[i + blockDim.x*2];
				dist = sqrt((Lx - Rx)*(Lx-Rx) + (Ly - Ry)*(Ly - Ry) + (Lz - Rz)*(Lz - Rz));
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
