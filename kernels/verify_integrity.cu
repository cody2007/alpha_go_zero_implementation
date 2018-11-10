#define ASSERT_S(COND) {if(!(COND)){ printf("assertion failure %s:%i\n", __FILE__, __LINE__); exit(1);}}
#ifdef CUDA_DEBUG
	#define VERIFY_BUFFER_INTEGRITY {if(verify_buffer_integrity() != 1){ printf("assertion failure %s:%i\n", __FILE__, __LINE__); exit(1); }}
	//#define VERIFY_BUFFER_INTEGRITY {printf("verifying %s\n", __FILE__); if(verify_buffer_integrity() != 1){ printf("assertion failure %s:%i\n", __FILE__, __LINE__); exit(1); }}
#else
	#define VERIFY_BUFFER_INTEGRITY 

#endif

char verify_buffer_integrity(){
	cudaError_t err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	
	if(return_device_buffers() != 1){
		printf("err returnning buffers %s:%i\n", __FILE__, __LINE__);
		return 0;
	}
	
	int coord;
	for(int game = 0; game < BATCH_SZ; game++){
		////////////////// map tests
		for(int x = 0; x < MAP_SZ_X; x++){
			for(int y = 0; y < MAP_SZ_Y; y++){
				coord = game*MAP_SZ + x*MAP_SZ_Y + y;
				
				ASSERT_S((board_cpu[coord] == 0) || (board_cpu[coord] == 1) ||
						(board_cpu[coord] == -1));
			} // y
		} // x
		
		////////// todo test stones are not surrounded
	}		
	return 1;
}
