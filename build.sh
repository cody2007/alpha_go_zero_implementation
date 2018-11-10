#export CC=gcc-6
#export CXX=g++-6
rm *.o
rm *.so
export TF_INC=TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
	--expt-relaxed-constexpr -arch=sm_52 -use_fast_math #-g -G -lineinfo
#sm_52
#-gencode=arch=compute_61,code=sm_61 
g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L$TF_LIB -ltensorflow_framework \
  -L/usr/local/cuda/lib64 -D_GLIBCXX_USE_CXX11_ABI=0 #-g #-O3

cd py_util
echo
echo ".................."
echo "building py_util" 
./build.sh
cd ..
