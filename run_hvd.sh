#!/bin/bash
# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

## Initialize our own variables:
#output_file=""
#verbose=0
np=2
cuda='0,1'
in_dir='../datasets/tisv_pickles/'
ckpt='ckpt'

show_help(){
    echo "./run_hvd.sh -n 2 -c 0,1 -i ../datasets/tisv_pickles -p ckpt"
}

while getopts "h?n:c:i:p:" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    n)  np=$OPTARG
        ;;
    c)  cuda=$OPTARG
        ;;
    i)  in_dir=$OPTARG
        ;;
    p)  ckpt=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

#echo "verbose=$verbose, output_file='$output_file', Leftovers: $@"
#/tmp/demo-getopts.sh -vf /etc/hosts foo bar
#output: verbose=1, output_file='/etc/hosts', Leftovers: foo bar
echo "mpirun -np $np \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH \
    -x LIBRARY_PATH \
    -x PYTHONPATH \
    -x PATH \
    -x CUDA_VISIBLE_DEVICES=$cuda \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_P2P_DISABLE=1 \
    --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0 \
    -mca pml ob1 -mca btl ^openib \ 
python estimator.py --in_dir $in_dir --ckpt $ckpt --use_horovod"

#-H 192.168.158.226:2 \
#-mca plm_rsh_args "-p 10029" \
mpirun -np $np \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH \
    -x LIBRARY_PATH \
    -x PYTHONPATH \
    -x PATH \
    -x CUDA_VISIBLE_DEVICES=$cuda \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_P2P_DISABLE=1 \
    --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0 \
    -mca pml ob1 -mca btl ^openib \
python estimator.py --in_dir $in_dir --ckpt $ckpt --use_horovod
