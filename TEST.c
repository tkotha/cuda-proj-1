



__global__ void PDH_kernel3(unsigned long long* d_histogram, 
							double* d_atom_x_list, 
							double* d_atom_y_list, 
							double* d_atom_z_list, 
							long long acnt, double res)
{

	//where shared memory size is 3*BLOCKSIZE*sizeof(double)
	extern __shared__ double R[];	
							
	int cur_id = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, h_pos;
	double  Lx, Ly, Lz, Rx, Ry, Rz;
	double dist;
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
		R[threadIdx.x] = Lx;
		R[threadIdx.x + blockDim.x] = Ly;
		R[threadIdx.x + blockDim.x*2] = Lz;
		__syncthreads();
		for(i = threadIdx.x+ 1; i < blockDim.x; i++)
		{
			cur_id = blockIdx.x * blockDim.x + i;	//we only proceed with valid threads for each thread i
			if(cur_id < acnt)
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