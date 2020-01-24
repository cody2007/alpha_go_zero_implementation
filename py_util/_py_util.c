#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "includes.h"

#include "rotate_reflect_imgs.c"
#include "init_tree.c"
#include "add_valid_mvs.c"
#include "register_mv.c"
#include "backup_visit.c"
#include "prune_tree.c"
#include "choose_moves.c"
#include "session_backup.c"
#include "return_tree.c"

static PyMethodDef py_util[] = {
	{"rotate_reflect_imgs", rotate_reflect_imgs, METH_VARARGS},
	{"init_tree", init_tree, METH_VARARGS},
	{"add_valid_mvs", add_valid_mvs, METH_VARARGS},
	{"register_mv", register_mv, METH_VARARGS},
	{"backup_visit", backup_visit, METH_VARARGS},
	{"prune_tree", prune_tree, METH_VARARGS},
	{"choose_moves", choose_moves, METH_VARARGS},
	{"session_backup", session_backup, METH_VARARGS},
	{"session_restore", session_restore, METH_VARARGS},
	{"return_tree", return_tree, METH_VARARGS},

	{NULL, NULL}
};

#if defined(_WIN32) || defined(_WIN64)
extern "C" void _declspec(dllexport) init_py_util(){
#else
extern void init_py_util(){
#endif
	srand(time(NULL));

	(void) Py_InitModule("_py_util", py_util);
	import_array();
	
}


