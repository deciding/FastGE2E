#!/bin/bash
# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

## Initialize our own variables:
#output_file=""
#verbose=0
generate=0
cuda=0
in_dir='../datasets/raw_vox/vox1/test/wav'
out_dir='spkid'
list_file='test_metas/veri_test2.txt'
ckpt='ckpt'

show_help(){
    echo "./run_test.sh [-g] -c 0 -i ../datasets/raw_vox/vox1/test/wav -o spkid -l test_metas//veri_test2.txt -p ckpt"
}

while getopts "h?gc:i:o:l:p:" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    g)  generate=1
        ;;
    c)  cuda=$OPTARG
        ;;
    i)  in_dir=$OPTARG
        ;;
    o)  out_dir=$OPTARG
        ;;
    l)  list_file=$OPTARG
        ;;
    p)  ckpt=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

if [ "$generate" == "1" ]; then
    python estimator.py --in_dir $in_dir --out_dir $out_dir --ckpt $ckpt --gpu_str $cuda --mode infer
fi

python npy_cmp.py --in_dir $out_dir --test_list $list_file
