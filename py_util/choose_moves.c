// choose maps based on tree search

/*	.Input("moving_player: int32") // [1]
	.Input("pol: float") // map, network's estimted probs
	.Input("CPUCT: float") // [1]

	.Output("to_coords: int32") // [BATCH_SZ]
	.Output("Q_map: float") // map
	.Output("P_map: float") // map
	.Output("visit_count_map: float") // map
*/
static PyObject *choose_moves(PyObject *self, PyObject *args){
	PyArrayObject *pol_np;
	float * pol, CPUCT;
	int moving_player, allow_pass_mv;

	if(!PyArg_ParseTuple(args, "iO!fi", &moving_player, &PyArray_Type, &pol_np, &CPUCT, &allow_pass_mv)) return NULL;
	
	/////////////////////// check inputs
	ASSERT(pol_np != NULL, "absent inputs")
	ASSERT(PyArray_TYPE(pol_np) == NPY_FLOAT32, "data type incorrect")
	ASSERT(PyArray_NDIM(pol_np) == 2, "dims must be 2")
	ASSERT(PyArray_STRIDE(pol_np, 1) == sizeof(pol[0]), "data not contigious or C-order")

	npy_intp * dims_in = PyArray_DIMS(pol_np);

	ASSERT(dims_in[0] == BATCH_SZ, "batch sz incorrect")
	ASSERT(dims_in[1] == ((MAP_SZ_X*MAP_SZ_Y) + 1), "map sz incorrect")

	pol = (float*) PyArray_DATA((PyArrayObject*) pol_np);

	///// output
	npy_intp dims[4];
	dims[0] = BATCH_SZ;
	dims[1] = (MAP_SZ_X*MAP_SZ_X) + 1;

	PyObject * to_coords_np = PyArray_SimpleNew(1, dims, NPY_INT32);
	PyObject * Q_map_np = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	PyObject * P_map_np = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	PyObject * visit_count_map_np = PyArray_SimpleNew(2, dims, NPY_FLOAT32);

	int * to_coords = (int *) PyArray_DATA((PyArrayObject*) to_coords_np);
	float * Q_map = (float *) PyArray_DATA((PyArrayObject*) Q_map_np);
	float * P_map = (float *) PyArray_DATA((PyArrayObject*) P_map_np);
	float * visit_count_map = (float *) PyArray_DATA((PyArrayObject*) visit_count_map_np);

	//////////////////////////////////////
	for(int gm = 0; gm < BATCH_SZ; gm++){

		////// init
		MAP_LOOP{
			int MO = gm*(MAP_SZ+1) + loc;
			P_map[MO] = 0;
			Q_map[MO] = 0;
			visit_count_map[MO] = 0;
		}
		
		CUR_TREE_INDS

		// pass move only valid move
		if(n_valid_mvs == 1){
			to_coords[gm] = MAP_SZ;
			continue;
		}

		// first move in the list is -1, the pass move, which is the last space in the maps (pol, P_map, Q_map)
		#define LOC_AND_MO int LOC = LO + mv_ind;\
				int map_loc = list_valid_mv_inds[LOC];\
				DASSERT(map_loc >= 0 && map_loc <= MAP_SZ);\
				int MO = gm*(MAP_SZ+1) + map_loc;

		/////////// sum all valid probs
		float prob_sum = 0;
		for(int mv_ind = 0; mv_ind < n_valid_mvs; mv_ind++){
			LOC_AND_MO
			prob_sum += pol[MO];
		}
		
		//////////// set prob value, compute tmp sums of Q & P
		int visit_sum = 0; // across mvs

		for(int mv_ind = 0; mv_ind < n_valid_mvs; mv_ind++){
			LOC_AND_MO
			
			// init move prob
			if(list_prob[LOC] == -1)
				list_prob[LOC] = pol[MO] / prob_sum;

			int visit_count_tmp = list_visit_count[LOC];
			if(visit_count_tmp == 0) visit_count_tmp = 1;

			// set maps
			Q_map[MO] = list_q_total[LOC] / visit_count_tmp;
			P_map[MO] = (CPUCT * list_prob[LOC]) / (1. + list_visit_count[LOC]);

			visit_sum += list_visit_count[LOC];

			visit_count_map[MO] = list_visit_count[LOC];
		}

		// compute U for each action, select max action
		float U_max = 0;
		int mv_ind_max = -1;
		float visit_sum_sqrt = sqrtf(visit_sum);
		for(int mv_ind = 0; mv_ind < n_valid_mvs; mv_ind++){
			LOC_AND_MO
			P_map[MO] *= visit_sum_sqrt;

			float U_tmp = Q_map[MO] + P_map[MO];
			if(((U_max < U_tmp) || (mv_ind_max == -1)) && (allow_pass_mv == 1 || 
					map_loc != MAP_SZ)){
				mv_ind_max = mv_ind;
				U_max = U_tmp;
			}
		}
		
		// set to_coords
		int LOC = LO + mv_ind_max;
		int map_loc = list_valid_mv_inds[LOC];
		DASSERT(map_loc >= 0 && map_loc <= MAP_SZ);
		to_coords[gm] = map_loc;

	} // gm

	/////////// return
	PyObject * ret = PyList_New(4);
	ASSERT(ret != 0, "err creating output list")

	ASSERT(PyList_SetItem(ret, 0, to_coords_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 1, Q_map_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 2, P_map_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 3, visit_count_map_np) == 0, "failed setting item");

	return ret;
}

