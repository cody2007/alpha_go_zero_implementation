static PyObject *session_backup(PyObject *self, PyObject *args){
	
	BMEM(tree_start2, tree_start, BATCH_SZ)

	Py_RETURN_NONE;
}

static PyObject *session_restore(PyObject *self, PyObject *args){
	
	BMEM(tree_start, tree_start2, BATCH_SZ)

	Py_RETURN_NONE;
}

