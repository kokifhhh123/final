module load mpi
module load cuda

/home/pp25/pp25s051/final/bin/main /home/pp25/pp25s051/final/testcase/building_small.png /home/pp25/pp25s051/final/output/building_small.png 36


/home/pp25/pp25s051/final/bin/main /home/pp25/pp25s051/final/testcase/flower.png /home/pp25/pp25s051/final/output/flower.png seq 36

/home/pp25/pp25s051/final/bin/main /home/pp25/pp25s051/final/testcase/building.png /home/pp25/pp25s051/final/output/building.png seq 36

k-means
1. sequential
2. openmp

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main /home/pp25/pp25s051/final/testcase/building.png /home/pp25/pp25s051/final/output/building.png seq 36

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main /home/pp25/pp25s051/final/testcase/building.png /home/pp25/pp25s051/final/output/building.png seq 36

slic
1. sequential
2. openmp



./kmeans input.png output.png mode K max_iters