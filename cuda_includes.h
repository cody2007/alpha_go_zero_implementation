#define CHECK_INIT { if(!op_initialized) init_op_launcher(); }
#define REQ_INIT ASSERT(op_initialized, "op not initialized")

#define CHECK_CUDA_ERR {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__); PANIC("");}}
#define CHECK_CUDA_ERR_R {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__); PANIC("");}}
#define MALLOC_ERR_CHECK {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__);  PANIC("");}}
#define MALLOC_ERR_CHECK_R {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__); PANIC("");}}

#ifdef CUDA_DEBUG
	#define DASSERT(A) assert(A);
#else
	#define DASSERT(A) 
#endif

#define BMEM(A, B, SZ) err = cudaMemcpy(A, B, SZ*sizeof(A[0]), cudaMemcpyDeviceToDevice);  MALLOC_ERR_CHECK
#define RMEM(A, B, SZ) err = cudaMemcpy(B, A, SZ*sizeof(A[0]), cudaMemcpyDeviceToDevice);  MALLOC_ERR_CHECK

char op_initialized; 

curandState_t* rand_states;

/////////////// game state
// [X]2 indicates backup variables used to restore session

char *board, *board2, board_cpu[BATCH_MAP_SZ];

// previous states to prevent ko
char *board_prev, *board_pprev;
char *board_prev2, *board_pprev2;

int * n_captures, *n_captures2; // [N_PLAYERS, BATCH_SZ]

int * ai_to_coord; // [BATCH_SZ], output of move_random_ai, input to move_unit

char * valid_mv_map_internal; // [BATCH_SZ, MAP_SZ], output of create_batch, input to move_unit
char * moved_internal; // [BATCH_SZ] used in move_random_ai, req. input to move_unit_launcher, results not used

// 1 or -1:
#define GET_PLAYER_VAL DASSERT((*moving_player == 0) || (*moving_player == 1)); char player_val = ((*moving_player == 0) * 2 )- 1;

#define CHK_VALID_MAP_COORD(COORD) DASSERT((COORD) >= 0 && (COORD) < MAP_SZ)
#define CHK_VALID_MV_MAP_COORD(COORD) DASSERT((COORD) >= 0 && (COORD) <= MAP_SZ)

// count valid mvs and store n_valid_mvs
#define COUNT_VALID \
	int n_valid_mvs = 0;\
	int valid_mv_inds[MAP_SZ];\
	MAP_LOOP{\
		if(valid_mv_map_internal[gm_offset + loc]){\
			valid_mv_inds[n_valid_mvs] = loc;\
			n_valid_mvs ++;\
		}\
	}\
	if(!n_valid_mvs) return; // no valid mvs

