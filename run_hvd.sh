#!/bin/bash
mpirun -np 2 \
    -H 192.168.158.226:2 \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH \
    -x LIBRARY_PATH \
    -x PYTHONPATH \
    -x PATH \
    -x CUDA_VISIBLE_DEVICES=4,5 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_P2P_DISABLE=1 \
    -mca plm_rsh_args "-p 10029" \
    --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0 \
    -mca pml ob1 -mca btl ^openib \
python estimator.py --in_dir ../datasets/tisv_pickles/ --ckpt tmp3 --use_horovod
