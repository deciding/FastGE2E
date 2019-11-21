# FastGE2E
##Train GE2E faster

##Command:
CUDA_VISIBLE_DEVICES=2,3 \
mpirun -np 1 \
    -H 192.168.158.226:1 \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH \
    -x LIBRARY_PATH \
    -x PYTHONPATH \
    -x PATH \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_P2P_DISABLE=1 \
    -mca plm_rsh_args "-p 10029" \
    --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0 \
    -mca pml ob1 -mca btl ^openib \
    -x HOROVOD_LOG_LEVEL=debug \
python estimator.py --in_dir ../datasets/tisv_pickles/ --ckpt tmp

##Troubleshooting:
Problem: hang

Possible reasons:
1. ssh auth
2. non-routable interfaces like: docker0, lo
3. problem of horovod

Checking process:
1. check ssh
2. check ifconfig
3. open the HOROVOD_LOG_LEVEL
4. use <command> hostname to have a simple check
5. ldd /home/kailingtang/.virtualenvs/tacotron/lib/python3.5/site-packages/horovod/tensorflow/mpi_lib.cpython-35m-x86_64-linux-gnu.so

Reason:
horovod is not installed properly

Solution:
1. pip uninstall horovod
2. HOROVOD_WITH_MPI=1 HOROVOD_CUDA_INCLUDE=/usr/local/cuda-9.0/targets/x86_64-linux/include/ HOROVOD_CUDA_LIB=/usr/local/cuda-9.0/targets/x86_64-linux/lib/ HOROVOD_CUDA_HOME=/usr/local/cuda-9.0/bin/ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod
3. check: ldd /home/kailingtang/.virtualenvs/tacotron/lib/python3.5/site-packages/horovod/tensorflow/mpi_lib.cpython-35m-x86_64-linux-gnu.so


How to check horovod gradient update is an aggregation?
"""
76     else:                                                                                                                  
77         with tf.device(device_dense):                                                                                      
78             horovod_size = tf.cast(size(), dtype=tensor.dtype)                                                             
79             tensor_compressed, ctx = compression.compress(tensor)                                                          
80             if "cell_1/lstm_cell/MatMul_1/Enter_grad/b_acc_3" in tensor_compressed.name:                                   
81                 opt_print=tf.print("=====tensor_compressed=====", tensor_compressed[0])                                    
82             summed_tensor_compressed = _allreduce(tensor_compressed)                                                       
83             summed_tensor = compression.decompress(summed_tensor_compressed, ctx)                                          
84             if "cell_1/lstm_cell/MatMul_1/Enter_grad/b_acc_3" in tensor_compressed.name:                                   
85                 opt_print2=tf.print("=====summed_tensor=====", summed_tensor[0])                                           
86                 with tf.control_dependencies([opt_print, opt_print2]):                                                     
87                     new_tensor = (summed_tensor / horovod_size) if average else summed_tensor                              
88             else:                                                                                                          
89                 new_tensor = (summed_tensor / horovod_size) if average else summed_tensor                                  
90         return new_tensor 
"""
