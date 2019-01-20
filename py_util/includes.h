#include "Python.h"
#include "arrayobject.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../includes.h"

#ifdef CUDA_DEBUG
	#define DASSERT(A) ASSERT(A, "assertion error")
#else
	#define DASSERT(A) 
#endif

#define TREE_BUFFER_SZ 500000 //  250000 //190000
#define MV_BUFFER_SZ 7000000 //5000000 //4500000 //4250000 //4000000 // 3760000


//#define TREE_BUFFER_SZ 19000//0 //70000//(1 200 000)//(800000)//*600000*2)
//#define MV_BUFFER_SZ 276000//0 //2 000 000 //1200000 //900000 //TREE_BUFFER_SZ

#define BMEM(A, B, SZ) memcpy(A, B, SZ*sizeof(A[0]));

//////////////////// tree
// create_batch: creates leaves (ex. list_valid_mv_inds)
// choose_moves: sets list_prob
// backup_visit: sets list_q_total
// move_unit: increments visit count, creates new tree node, sets tree_parent

////// node information:
unsigned tree_sz[BATCH_SZ];
unsigned tree_start[BATCH_SZ], tree_start2[BATCH_SZ]; // tree_sz2: for session backup/restoration

#define B_TREE_SZ (BATCH_SZ * TREE_BUFFER_SZ)
char tree_player[B_TREE_SZ], tree_player_back[TREE_BUFFER_SZ];
int tree_parent[B_TREE_SZ], tree_parent_back[TREE_BUFFER_SZ];

// start index for list_valid_mv_inds, list_valid_tree_inds:
int tree_list_sz[B_TREE_SZ], tree_list_sz_back[TREE_BUFFER_SZ];
int tree_list_start[B_TREE_SZ], tree_list_start_back[TREE_BUFFER_SZ];

////// lists (leaf information)
#define B_MV_SZ (BATCH_SZ * MV_BUFFER_SZ)
unsigned list_sz[BATCH_SZ];
short list_valid_mv_inds[B_MV_SZ], list_valid_mv_inds_back[MV_BUFFER_SZ]; // (first entry is always the pass mv)
int list_valid_tree_inds[B_MV_SZ], list_valid_tree_inds_back[MV_BUFFER_SZ];
float list_q_total[B_MV_SZ], list_q_total_back[MV_BUFFER_SZ];
float list_prob[B_MV_SZ], list_prob_back[MV_BUFFER_SZ];
unsigned list_visit_count[B_MV_SZ], list_visit_count_back[MV_BUFFER_SZ];

// used in prune tree:
unsigned tree_cp_old_stack[TREE_BUFFER_SZ], tree_cp_new_stack[TREE_BUFFER_SZ]; // prune_tree, tree inds to cp

////////////////////////////////////////
#define CHK_T_IND DASSERT(tree_sz[gm] < TREE_BUFFER_SZ);\
	  	DASSERT(t_ind >= 0 && t_ind < tree_sz[gm]);

#define CHK_L_IND DASSERT(list_sz[gm] < MV_BUFFER_SZ);\
 		DASSERT(l_ind >= 0)\
		DASSERT(l_ind < list_sz[gm])


#define CHK_N_VALID_MVS DASSERT(n_valid_mvs > 0 && n_valid_mvs <= (MAP_SZ+1));\
			DASSERT( (n_valid_mvs + tree_list_start[TO]) <= list_sz[gm]);

#define TO_FRM_T_IND CHK_T_IND; TO = gm*TREE_BUFFER_SZ + t_ind;
#define LO_FRM_L_IND CHK_L_IND; LO = gm*MV_BUFFER_SZ + l_ind;

#define CUR_TREE_INDS_WO_MV_CHK int TO, LO;\
	int t_ind = tree_start[gm]; TO_FRM_T_IND\
	int l_ind = tree_list_start[TO]; LO_FRM_L_IND\
	int n_valid_mvs = tree_list_sz[TO]; 

#define CUR_TREE_INDS CUR_TREE_INDS_WO_MV_CHK \
		      CHK_N_VALID_MVS


