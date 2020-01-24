CUDA_PATH=/usr/local/cuda/lib64

#export CC=gcc-6
#export CXX=g++-6
rm *.o
rm *.so
export TF_INC=TF_INC=$(python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_INC=$(python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#nvcc -std=c++11 -c -o cuda_op_kernel_52.cu.o cuda_op_kernel.cu.cc ${TF_FLAGS[@]} -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
#	--expt-relaxed-constexpr -arch=sm_52 -use_fast_math #-g -G -lineinfo
#nvcc -std=c++11 -c -o cuda_op_kernel_75.cu.o cuda_op_kernel.cu.cc ${TF_FLAGS[@]} -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
#	--expt-relaxed-constexpr -arch=sm_75 -use_fast_math #-g -G -lineinfo

#sm_52
#-gencode=arch=compute_61,code=sm_61 
TF_CFLAGS=( $(python2 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python2 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o cuda_op_kernel_52.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr
nvcc -std=c++11 -c -o cuda_op_kernel_75.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr
 

g++ -std=c++11 -shared -o cuda_op_kernel_52.so cuda_op_kernel.cc \
  cuda_op_kernel_52.cu.o ${TF_CFLAGS[@]} -fPIC -L $CUDA_PATH -lcudart ${TF_LFLAGS[@]}
	
g++ -std=c++11 -shared -o cuda_op_kernel_75.so cuda_op_kernel.cc \
  cuda_op_kernel_75.cu.o ${TF_CFLAGS[@]} -fPIC -L $CUDA_PATH -lcudart ${TF_LFLAGS[@]}

#g++ -std=c++11 -shared -o cuda_op_kernel_52.so cuda_op_kernel.cc \
#  cuda_op_kernel_52.cu.o ${TF_FLAGS[@]} -fPIC -lcudart -L$TF_LIB  \
#  -L/usr/local/cuda/lib64 -I $TF_INC -I$TF_INC/external/nsync/public -D_GLIBCXX_USE_CXX11_ABI=1 ${TF_FLAGS[@]} #-g #-O3
#g++ -std=c++11 -shared -o cuda_op_kernel_75.so cuda_op_kernel.cc \
#  cuda_op_kernel_75.cu.o ${TF_FLATS[@]} -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L$TF_LIB  \
#  -L/usr/local/cuda/lib64 -D_GLIBCXX_USE_CXX11_ABI=1 ${TF_FLAGS[@]} -D GOOGLE_CUDA=1 #-g #-O3

#g++ -std=c++11 -shared -o cuda_op_kernel_52.so cuda_op_kernel.cc \
#  cuda_op_kernel_52.cu.o ${TF_FLAGS[@]} -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L$TF_LIB  \
#  -L/usr/local/cuda/lib64 -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_FLAGS[@]} #-g #-O3
#g++ -std=c++11 -shared -o cuda_op_kernel_75.so cuda_op_kernel.cc \
#  cuda_op_kernel_75.cu.o ${TF_FLAGS[@]} -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L$TF_LIB  \
#  -L/usr/local/cuda/lib64 -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_FLAGS[@]} #-g #-O3



cd py_util
echo
echo ".................."
echo "building py_util" 
./build_centos.sh
cd ..
