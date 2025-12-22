
module load cuda
make clean
make



# seq
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output/kmeans_jellyfish_seq.png kmeans seq 3
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output/slic_jellyfish_seq.png slic seq 5000


# omp
srun -p nvidia -N1 -n2 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output/kmeans_jellyfish_omp.png kmeans omp 3
srun -p nvidia -N1 -n2 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output/slic_jellyfish_omp.png slic omp 5000


# cuda
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output/kmeans_jellyfish_cuda_opt.png kmeans cuda_opt 3
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output/slic_jellyfish_cuda.png slic cuda 5000


# cuda opt
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output/kmeans_jellyfish_cuda_opt_warp.png kmeans cuda_opt_warp 3
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output/slic_jellyfish_cudaopt.png slic cuda_opt 5000