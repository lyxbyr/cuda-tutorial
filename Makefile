00:
	@nvcc 00add/00add.cu -o exe0
	@nvcc 01simple_kernel.cu -o exe1
	@nvcc 01simple_kernel2.cu -o exe1_2
	@nvcc 02add_kernel.cu -o exe2
	@nvcc 03add_kernel.cu -o exe3	
	@nvcc 04error_handler/04error_handler.cu -o exe4