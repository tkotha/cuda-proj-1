In this folder should be the SDH.cu file, the pdf report, and this readme.

to compile the SDH.cu file, first copy the contents of this folder to the c4cudaNN machines, using the instructions provided on canvas.

Once it's there, and you have issued:
module load apps/cuda/7.5

go to the directory where you copied the contents, and issue:
nvcc SDH.cu -o SDH

once there is an SDH executable, run it with:
./SDH {#of_samples} {bucket_width} {block_size}

where:
		#of_samples is an integer literal
		bucket_width is a double literal
		block_size is an integer literal

The GPU timing results will appear after execution of the kernel.


Note: After much experimentation, we've noticed that a block_size of 32 runs the fastest, so please run the program with that parameter set.
