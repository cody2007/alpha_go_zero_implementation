#define ADD_NODE_TO_STACK(NODE) \
	tree_cp_old_stack[stack_sz] = NODE;\
	tree_cp_new_stack[stack_sz] = tree_sz_back;\
	\
	stack_sz ++;\
	tree_sz_back ++;\
	DASSERT(stack_sz < TREE_BUFFER_SZ)\
	DASSERT(tree_sz_back < TREE_BUFFER_SZ)\
	DASSERT(tree_sz_back <= tree_sz[gm])

#define CHK_PREV_TREE_IND(IND, SZ) DASSERT((int)IND >= 0 && IND < SZ) 

static PyObject *prune_tree(PyObject *self, PyObject *args){
	int single_game;

	if(!PyArg_ParseTuple(args, "i", &single_game)) return NULL;

	// only prune first game, reset everything else
	int games_loop = BATCH_SZ;
	if(single_game == 1){
		games_loop = 1;
		
		memset(&tree_start[1], 0, sizeof(tree_start[0])*(BATCH_SZ-1));
		memset(&list_sz[1], 0, sizeof(list_sz[0])*(BATCH_SZ-1));

		for(int i = 1; i < BATCH_SZ; i++){
			tree_sz[i] = 1;
			tree_list_start[i*TREE_BUFFER_SZ] = -1;
			tree_parent[i*TREE_BUFFER_SZ] = -1;
		}
	}

	for(int gm = 0; gm < games_loop; gm++){
		int TOFF = gm*TREE_BUFFER_SZ;
		int LOFF = gm*MV_BUFFER_SZ;

		int stack_sz = 0;
		int tree_sz_back = 0;
		int list_sz_back = 0;
	
		DASSERT(tree_sz[gm] < TREE_BUFFER_SZ)
		DASSERT(list_sz[gm] < MV_BUFFER_SZ)

		//////////////////
		// start from tree_start[gm] and mv forward keeping all leaves
		ADD_NODE_TO_STACK(tree_start[gm])
		
		for(int stack_loc = 0; stack_loc < stack_sz; stack_loc++){
			CHK_PREV_TREE_IND(tree_cp_new_stack[stack_loc], tree_sz_back)
			CHK_PREV_TREE_IND(tree_cp_old_stack[stack_loc], tree_sz[gm])
			
			int TO_NEW = tree_cp_new_stack[stack_loc];
			int TO = TOFF + tree_cp_old_stack[stack_loc];
		
			///////////// cp node
			tree_player_back[TO_NEW] = tree_player[TO];
			tree_list_sz_back[TO_NEW] = tree_list_sz[TO];

			if(tree_list_start[TO] != -1) // new list slot
				tree_list_start_back[TO_NEW] = list_sz_back;
			else
				tree_list_start_back[TO_NEW] = -1;

			/////////////////////// set tree_parent

			// parent of new root is non-existant
			if(tree_cp_old_stack[stack_loc] == tree_start[gm]){
				tree_parent_back[TO_NEW] = -1;
			}else{
				// find new tree_parent index
				char found = 0; int stack_loc_j;
				for(stack_loc_j = 0; stack_loc_j < stack_sz; stack_loc_j++){
					if(tree_cp_old_stack[stack_loc_j] != tree_parent[TO])
						continue;
					found = 1;
					break;
				}
				assert(found == 1);
				
				tree_parent_back[TO_NEW] = tree_cp_new_stack[stack_loc_j];
			}

			/////////////// cp list
			DASSERT(tree_list_sz[TO] <= (MAP_SZ+1))
			DASSERT(tree_list_sz[TO] >= 0)
			DASSERT((tree_list_start_back[TO_NEW] >= 0 && tree_list_start_back[TO_NEW] <= list_sz_back) || tree_list_start_back[TO_NEW] == -1)

			for(int mv_ind = 0; mv_ind < tree_list_sz[TO]; mv_ind++){
				int LO = LOFF + tree_list_start[TO] + mv_ind;
				int LO_NEW = tree_list_start_back[TO_NEW] + mv_ind;

				// cp list
				list_valid_mv_inds_back[LO_NEW] = list_valid_mv_inds[LO];
				list_q_total_back[LO_NEW] = list_q_total[LO];
				list_prob_back[LO_NEW] = list_prob[LO];
				list_visit_count_back[LO_NEW] = list_visit_count[LO];

				// tree node to copy
				if(list_valid_tree_inds[LO] != -1){
					list_valid_tree_inds_back[LO_NEW] = tree_sz_back;

					ADD_NODE_TO_STACK(list_valid_tree_inds[LO])
				}else
					list_valid_tree_inds_back[LO_NEW] = -1;

				list_sz_back ++;
				DASSERT(list_sz_back <= list_sz[gm])
			}
		}			
	
		/////////// copy over
		tree_start[gm] = 0;
		tree_sz[gm] = tree_sz_back;
		list_sz[gm] = list_sz_back;

		DASSERT((tree_sz[gm] < TREE_BUFFER_SZ) && (tree_sz[gm] > 0))
		BMEM2(&tree_player[TOFF], tree_player_back, tree_sz[gm])
		BMEM2(&tree_parent[TOFF], tree_parent_back, tree_sz[gm])
		BMEM2(&tree_list_sz[TOFF], tree_list_sz_back, tree_sz[gm])
		BMEM2(&tree_list_start[TOFF], tree_list_start_back, tree_sz[gm])

		DASSERT(list_sz[gm] < MV_BUFFER_SZ)
		BMEM2(&list_valid_mv_inds[LOFF], list_valid_mv_inds_back, list_sz[gm])
		BMEM2(&list_valid_tree_inds[LOFF], list_valid_tree_inds_back, list_sz[gm])
		BMEM2(&list_q_total[LOFF], list_q_total_back, list_sz[gm])
		BMEM2(&list_prob[LOFF], list_prob_back, list_sz[gm])
		BMEM2(&list_visit_count[LOFF], list_visit_count_back, list_sz[gm])
	} // gm

	Py_RETURN_NONE;
}

