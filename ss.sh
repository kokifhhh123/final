

srun -p nvidia -N1 -n1 --gres=gpu:1 ../bin/main ../testcase/jellyfish.png ../output2/slic_jellyfish_cuda.png slic cuda 5000
