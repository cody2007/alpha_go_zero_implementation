#define ADD_NODE_TO_STACK(NODE) \
	tree_cp_old_stack[stack_sz] = NODE;\
	tree_cp_new_stack[stack_sz] = tree_sz[gm];\
	\
	stack_sz ++;\
	tree_sz[gm] ++;\
	DASSERT(stack_sz < TREE_BUFFER_SZ)\
	DASSERT(tree_sz[gm] < TREE_BUFFER_SZ)

#define CHK_PREV_TREE_IND(IND) DASSERT((int)IND >= 0 && IND < prev_tree_sz) 

#define CHK_LIST DASSERT(tree_list_sz[TO_NEW] <= (MAP_SZ+1))\
		DASSERT(tree_list_sz[TO_NEW] == -1 || tree_list_sz[TO_NEW] > 0)\
		DASSERT((tree_list_start_back[TO] >= 0 && tree_list_start_back[TO] < MV_BUFFER_SZ) ||  \
			tree_list_start_back[TO] == -1)

#define TREE_IND_PLACEHOLDER 0 

static PyObject *prune_tree(PyObject *self, PyObject *args){

	for(int gm = 0; gm < BATCH_SZ; gm++){
		int TOFF = gm*TREE_BUFFER_SZ;
		int LOFF = gm*MV_BUFFER_SZ;

		// backup
		BMEM(tree_player_back, &tree_player[TOFF], TREE_BUFFER_SZ)
		BMEM(tree_parent_back, &tree_parent[TOFF], TREE_BUFFER_SZ)
		BMEM(tree_list_sz_back, &tree_list_sz[TOFF], TREE_BUFFER_SZ)
		BMEM(tree_list_start_back, &tree_list_start[TOFF], TREE_BUFFER_SZ)

		BMEM(list_valid_mv_inds_back, &list_valid_mv_inds[LOFF], MV_BUFFER_SZ)
		BMEM(list_valid_tree_inds_back, &list_valid_tree_inds[LOFF], MV_BUFFER_SZ)
		BMEM(list_q_total_back, &list_q_total[LOFF], MV_BUFFER_SZ)
		BMEM(list_prob_back, &list_prob[LOFF], MV_BUFFER_SZ)
		BMEM(list_visit_count_back, &list_visit_count[LOFF], MV_BUFFER_SZ)

		////////// re-initialize
		memset(&tree_player[TOFF], 0, TREE_BUFFER_SZ*sizeof(tree_player[0]));
		memset(&list_q_total[LOFF], 0, MV_BUFFER_SZ*sizeof(list_q_total[0]));
		memset(&list_visit_count[LOFF], 0, MV_BUFFER_SZ*sizeof(list_visit_count[0]));

		for(int i = TOFF; i < (TOFF + TREE_BUFFER_SZ); i++){
			tree_parent[i] = -1;
			tree_list_start[i] = -1;
			tree_list_sz[i] = -1;
		}

		for(int i = LOFF; i < (LOFF + MV_BUFFER_SZ); i++){
			list_valid_mv_inds[i] = -1;
			list_valid_tree_inds[i] = -1;
			list_prob[i] = -1;
		}
		/////

		int stack_sz = 0;
		
		#ifdef CUDA_DEBUG
		unsigned prev_tree_sz = tree_sz[gm];
		#endif
		unsigned cur_tree_start = tree_start[gm];
		
		tree_sz[gm] = 0;
		list_sz[gm] = 0;

		////////////////////
		// start from cur_tree_start and mv backward keeping only the parents (back to the root)
		// first entry is parent of current node, second is parent of parent, ...

		int tree_start_parent = tree_parent_back[cur_tree_start];
		CHK_PREV_TREE_IND(tree_start_parent)

		int prev_node = 0;
		while(1){
			int TO_NEW = TOFF + tree_sz[gm];
			int TO = tree_start_parent;

			tree_sz[gm] ++;
			DASSERT(tree_sz[gm] < TREE_BUFFER_SZ)

			///////////////// cp node
			tree_player[TO_NEW] = tree_player_back[TO];
			tree_list_sz[TO_NEW] = tree_list_sz_back[TO];
			tree_list_start[TO_NEW] = list_sz[gm];

			/////////////// cp list
			#ifdef CUDA_DEBUG
				char found = 0;
				#define SET_FOUND found = 1;
			#else
				#define SET_FOUND
			#endif
			CHK_LIST
			for(int mv_ind = 0; mv_ind < tree_list_sz[TO_NEW]; mv_ind++){
				int LO = tree_list_start_back[TO] + mv_ind;
				int LO_NEW = LOFF + tree_list_start[TO_NEW] + mv_ind;

				// cp list
				list_valid_mv_inds[LO_NEW] = list_valid_mv_inds_back[LO];
				list_q_total[LO_NEW] = list_q_total_back[LO];
				list_prob[LO_NEW] = list_prob_back[LO];
				list_visit_count[LO_NEW] = list_visit_count_back[LO];

				//////// tree ind

				// from current position, insert placeholder
				if(list_valid_tree_inds_back[LO] == cur_tree_start){
					list_valid_tree_inds[LO_NEW] = TREE_IND_PLACEHOLDER;
					DASSERT(tree_sz[gm] == 1) // nothing should come before
					SET_FOUND

				// prev entry just inserted
				}else if(list_valid_tree_inds_back[LO] == prev_node){
					list_valid_tree_inds[LO_NEW] = tree_sz[gm] - 2;
					DASSERT(list_valid_tree_inds[LO_NEW] >= 0)
					SET_FOUND

				// set as uninitialized, not nodes that we'll keep
				}else
					list_valid_tree_inds[LO_NEW] = -1;

				LO_NEW ++;
				list_sz[gm] ++;
				DASSERT(list_sz[gm] < MV_BUFFER_SZ)
			}
		
			#ifdef CUDA_DEBUG
				if(found == 0 && gm == 0){
					printf("tree_sz %i list_sz %i tree_start_parent %i tree_list_sz %i\n", tree_sz[gm], list_sz[gm], tree_start_parent,
						tree_list_sz[TO_NEW]);
					for(int mv_ind = 0; mv_ind < tree_list_sz[TO_NEW]; mv_ind++){
						int LO = LOFF + tree_list_start_back[TO] + mv_ind;
						printf("list_valid_tree_inds_back[%i] %i\n", mv_ind, list_valid_tree_inds_back[LO]);
					}
					assert(0);
				}
			#endif
			DASSERT(found == 1)

			/////////////////////// set tree_parent
			
			if(tree_parent_back[TO] == -1){ // at root
				tree_parent[TO_NEW] = -1;
				break;
			}

			// prepare for next entry
			tree_parent[TO_NEW] = tree_sz[gm];
			prev_node = tree_start_parent;
			tree_start_parent = tree_parent_back[TO];
			CHK_PREV_TREE_IND(tree_start_parent)
		}

		///////////// replace placeholder tree ind with current tree sz
		char found = 0;
		int n_mvs = tree_list_sz[TOFF]; // first entry is parent of current location
		DASSERT(tree_list_start[TOFF] == 0)

		for(int mv_ind = 0; mv_ind < n_mvs; mv_ind++){
			int LO_NEW = LOFF + mv_ind; // first entry is parent of current location
			
			DASSERT(list_valid_tree_inds[LO_NEW] == -1 || list_valid_tree_inds[LO_NEW] == TREE_IND_PLACEHOLDER)
			if(list_valid_tree_inds[LO_NEW] != TREE_IND_PLACEHOLDER)
				continue;
			
			list_valid_tree_inds[LO_NEW] = tree_sz[gm];
			found = 1;
			break;
		}
		assert(found == 1);
		tree_start[gm] = tree_sz[gm]; // set start as current position

		//////////////////
		// start from cur_tree_start and mv forward keeping all leaves
		ADD_NODE_TO_STACK(cur_tree_start)
		
		for(int stack_loc = 0; stack_loc < stack_sz; stack_loc++){
			DASSERT(tree_cp_new_stack[stack_loc] < tree_sz[gm]);
			CHK_PREV_TREE_IND(tree_cp_old_stack[stack_loc])
			
			int TO_NEW = TOFF + tree_cp_new_stack[stack_loc];
			int TO = tree_cp_old_stack[stack_loc];
		
			///////////// cp node
			tree_player[TO_NEW] = tree_player_back[TO];
			tree_list_sz[TO_NEW] = tree_list_sz_back[TO];

			if(tree_list_start_back[TO] != -1) // new list slot
				tree_list_start[TO_NEW] = list_sz[gm];
			else
				tree_list_start[TO_NEW] = -1;

			/////////////////////// set tree_parent

			// parent of new root is non-existant
			if(tree_cp_new_stack[stack_loc] == tree_start[gm]){
				tree_parent[TO_NEW] = 0; // first entry
			}else{
				// find new tree_parent index
				char found = 0; int stack_loc_j;
				for(stack_loc_j = 0; stack_loc_j < stack_sz; stack_loc_j++){
					if(tree_cp_old_stack[stack_loc_j] != tree_parent_back[TO])
						continue;
					found = 1;
					break;
				}
				assert(found == 1);
				
				tree_parent[TO_NEW] = tree_cp_new_stack[stack_loc_j];
			}

			/////////////// cp list
			CHK_LIST
			for(int mv_ind = 0; mv_ind < tree_list_sz[TO_NEW]; mv_ind++){
				int LO = tree_list_start_back[TO] + mv_ind;
				int LO_NEW = LOFF + tree_list_start[TO_NEW] + mv_ind;

				// cp list
				list_valid_mv_inds[LO_NEW] = list_valid_mv_inds_back[LO];
				list_q_total[LO_NEW] = list_q_total_back[LO];
				list_prob[LO_NEW] = list_prob_back[LO];
				list_visit_count[LO_NEW] = list_visit_count_back[LO];

				// tree node to copy
				if(list_valid_tree_inds_back[LO] != -1){
					list_valid_tree_inds[LO_NEW] = tree_sz[gm];

					ADD_NODE_TO_STACK(list_valid_tree_inds_back[LO])
				}else
					list_valid_tree_inds[LO_NEW] = -1;

				LO_NEW ++;
				list_sz[gm] ++;
				DASSERT(list_sz[gm] < MV_BUFFER_SZ)
			}
		}			
		
	} // gm

	Py_RETURN_NONE;
}

