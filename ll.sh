K=5000
algo=slic
name=jellyfish
type=cudaopt


# srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/${name}.png ../output/${name}_${algo}_${type}_${K}.png ${algo} ${type} ${K}

# bash ../ll.sh


# srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output2/kmeans_flower_cuda.png kmeans cuda 128
# srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output2/kmeans_flower_cuda_opt.png kmeans cuda_opt 128
# srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output2/kmeans_flower_cuda_warp.png kmeans cuda_opt_warp 128

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output2/slic_jellyfish_cuda.png slic cuda 5000
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output2/slic_jellyfish_cuda_opt.png slic cuda_opt 5000

