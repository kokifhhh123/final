


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


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/floor.png ../output/floor_kmeans.png kmeans cuda 8

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/leaves.png ../output/leaves.png kmeans cuda 4


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/bb.png ../output/bb.png kmeans cuda 4

slic
1. sequential
2. openmp




./kmeans input.png output.png algo mode K max_iters

algo kmeans/slic

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_slic_seq.png slic seq 10000

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_slic_omp.png slic omp 10000

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_slic_cuda.png slic cuda 10000

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/building.png ../output/building_slic_cuda.png slic cuda 10000


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/building.png ../output/building_slic_cudaopt.png slic cudaopt 10000
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/flower.png ../output/flower_slic_cudaopt.png slic cudaopt 10000


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/lion.png ../output/lion_slic_cuda.png slic cuda 10000
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/lion.png ../output/lion_slic_cudaopt.png slic cudaopt 10000
srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/lion.png ../output/lion_slic_cudaopt_15000.png slic cudaopt 15000


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/chair.png ../output/chair_slic_cudaopt_1500.png slic cudaopt 1500

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/leaves.png ../output/_slic.png slic cuda 700

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/bb.png ../output/bb_slic.png slic cudaopt 500


srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/kk.png ../output/kk_slic.png slic cudaopt 700

/home/pp25/pp25s051/final/testcase/lion.png