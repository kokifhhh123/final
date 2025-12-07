


module load mpi
module load cuda

/home/pp25/pp25s051/final/bin/main /home/pp25/pp25s051/final/testcase/building_small.png /home/pp25/pp25s051/final/output/building_small.png 36


/home/pp25/pp25s051/final/bin/main /home/pp25/pp25s051/final/testcase/flower.png /home/pp25/pp25s051/final/output/flower.png seq 36

/home/pp25/pp25s051/final/bin/main /home/pp25/pp25s051/final/testcase/building.png /home/pp25/pp25s051/final/output/building.png seq 36

k-means
1. sequential
2. openmp

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main /home/pp25/pp25s051/final/testcase/building.png /home/pp25/pp25s051/final/output/building.png seq 36



srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/building.png ../output/building.png kmeans seq 128


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/building.png ../output/building_cuda.png kmeans cuda 3

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/building.png ../output/building_cuda_128.png kmeans cuda 128


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/building.png ../output/building_cuda_opt.png kmeans cuda_opt 128
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_cuda_opt.png kmeans cuda_opt 128


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/building.png ../output/building_cuda_opt.png kmeans cuda_opt_more 128
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_cuda_opt.png kmeans cuda_opt_more 16


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/building.png ../output/building_cuda_warp.png kmeans cuda_opt_warp 128
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_cuda_warp.png kmeans cuda_opt_warp 3


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_cuda_soa.png kmeans cuda_opt_soa 16

slic
1. sequential
2. openmp




./kmeans input.png output.png algo mode K max_iters

algo kmeans/slic

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_slic.png slic seq 1500