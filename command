
srun -p nvidia -N1 -n1 --gres=gpu:1 ./hw3-2 /home/pp25/pp25s051/share/hw3/testcases/c20.1 ./outc20.1
srun -p nvidia -N1 -n1 --gres=gpu:1 ./hw3-2 /home/pp25/pp25s051/share/hw3/testcases/p11k1 ./p11k1
srun -p nvidia -N1 -n1 --gres=gpu:1 ./hw3-2 /home/pp25/pp25s051/share/hw3/testcases/p15k1 ./p15k1
srun -p nvidia -N1 -n1 --gres=gpu:1 ./hw3-2 /home/pp25/pp25s051/share/hw3/testcases/p20k1 ./p20k1
srun -p nvidia -N1 -n1 --gres=gpu:1 ./hw3-2 /home/pp25/pp25s051/share/hw3/testcases/p25k1 ./p25k1

srun -p nvidia -N1 -n1 --gres=gpu:2 ./hw3-3 /home/pp25/pp25s051/share/hw3/testcases/p33k1 ./outp33k1


srun -n 1 --gres=gpu:1 /home/pp25/pp25s051/deviceQuery/deviceQuery

srun -N1 -n1 ./hw3-1 /home/pp25/pp25s051/share/hw3/testcases/c21.1 ./outc21.1



srun -p nvidia -N1 -n1 --gres=gpu:1 ./hw3-2 /home/pp25/pp25s051/share/hw3/testcases/c20.1 ./outc20.1


srun -p amd -N1 -n1 --gres=gpu:1 ./hw3-2-amd /home/pp25/pp25s051/share/hw3/testcases/c21.1 ./outc21.1

srun -p amd -N1 -n1 --gres=gpu:1 rocprof --stats ./hw3-2-amd /home/pp25/pp25s051/share/hw3/testcases/c21.1 ./outc21.1


srun -p amd -N1 -n1 --gres=gpu:1 \
  rocprof --stats -o stats_bsz64.csv \
  ./hw3-2-amd ./hw3-2-amd /home/pp25/pp25s051/share/hw3/testcases/c21.1 ./outc21.1


srun -p amd -N1 -n1 --gres=gpu:1 \
  rocprof --stats -o stats_bsz32.csv \
  ./hw3-2-amd ./hw3-2-amd /home/pp25/pp25s051/share/hw3/testcases/c21.1 ./outc21.1


srun -p amd -N1 -n1 --gres=gpu:1 \
  rocprof --stats -o stats_bsz64.csv \
  ./hw3-2-amd ./hw3-2-amd /home/pp25/pp25s051/share/hw3/testcases/c21.1 ./outc21.1



srun -p amd -N1 -n1 --gres=gpu:1 ./hw3-2-amd /home/pp25/pp25s051/share/hw3/testcases/p26k1 p26k1


srun -p amd -N1 -n1 --gres=gpu:1 ./hw4-amd /home/pp25/pp25s051/share/hw4/testcases-amd/t01 out



srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics gld_throughput 

/home/pp25/pp25s051/share/hw4/testcases/t25


srun -p nvidia -N1 -n1 --gres=gpu:1 \
nvprof --metrics \
achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput \
./hw4 /home/pp25/share/hw4/testcases/t25 t25.out


srun -p nvidia -N1 -n1 --gres=gpu:1 ./hw4 /home/pp25/share/hw4/testcases/t25 t25.out



srun -p amd -N1 -n1 --gres=gpu:1 \
  rocprof --stats  \
  ./hw4-amd /home/pp25/share/hw4/testcases/t25 t25.out


srun -p amd -N1 -n1 --gres=gpu:1 \
  ./hw4-amd /home/pp25/share/hw4/testcases/t25 t25.out

srun -p amd -N1 -n1 --gres=gpu:1 \
  ./hw4-amd /home/pp25/share/hw4/testcases/t30 t30.out

srun -p amd -N1 -n1 --gres=gpu:1 \
  ./hw4-amd /home/pp25/share/hw4/testcases/t15 t15.out

srun -p nvidia -N1 -n1 --gres=gpu:1 \
  nvprof \
  ./hw4 /home/pp25/share/hw4/testcases/t25 t25.out

# rocprof --timestamp on --hip-trace --hsa-trace ./a.out input.bin output.bin