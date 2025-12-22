



# srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output2/kmeans_flower_cuda.png kmeans cuda 128
# srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output2/kmeans_flower_cuda_opt.png kmeans cuda_opt 128
# srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output2/kmeans_flower_cuda_warp.png kmeans cuda_opt_warp 128


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output2/flower_slic_opt.png slic cuda 10000
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output2/flower_slic_opt.png slic cuda_opt 10000

# bash ../sss.sh