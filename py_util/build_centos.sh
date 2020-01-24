gcc _py_util.c -fPIC -O3 -I/usr/include/python2.7 -I/usr/include/numpy -I/usr/lib64/python2.7/site-packages/numpy/core/include/numpy -lpython2.7 -shared -o _py_util.so -Wall

