#define CP_MAP(MAP) {   if(op == RETURN_VARS)\
				err = cudaMemcpy(outputs, MAP, BATCH_MAP_SZ*sizeof(MAP[0]), cudaMemcpyDeviceToDevice);\
			else\
				err = cudaMemcpy(MAP, outputs, BATCH_MAP_SZ*sizeof(MAP[0]), cudaMemcpyDeviceToDevice);\
		MALLOC_ERR_CHECK}

#define CP_MAP_DT(MAP, dt) {   if(op == RETURN_VARS)\
					err = cudaMemcpy(outputs, MAP, BATCH_MAP_SZ*sizeof(dt), cudaMemcpyDeviceToDevice);\
				else\
					err = cudaMemcpy(MAP, outputs, BATCH_MAP_SZ*sizeof(dt), cudaMemcpyDeviceToDevice);\
		MALLOC_ERR_CHECK}

#define CP_DT(MAP, SZ, dt) {    if(op == RETURN_VARS)\
					err = cudaMemcpy(outputs, MAP, SZ*sizeof(dt), cudaMemcpyDeviceToDevice);\
				else\
					err = cudaMemcpy(MAP, outputs, SZ*sizeof(dt), cudaMemcpyDeviceToDevice);\
			MALLOC_ERR_CHECK}


void vars_launcher(int var_idx, void * outputs, char op){
	REQ_INIT
	cudaError_t err;
	if(var_idx == BOARD_IDX) CP_MAP(board)
	else if(var_idx == VALID_MV_MAP_INTERNAL_IDX) CP_MAP(valid_mv_map_internal)
	else PANIC("unknown var_idx, return_vars_launcher");

}

