#!/bin/bash

solver="fiasco"
prot_in="$1.in.fiasco"
prot_out="out.pdb"
domain_size=100
max_sols=10000
timelim=20
idx=0
prot="proteins/1ZDD.in.fiasco"
idx=$[$idx+1]
for vside in 1
do
for kmax in 1000
do
#-----------------------------------------
   ./$solver \
    --input ${prot} \
    --outfile $prot_out \
    --domain-size $domain_size \
    --ensembles $max_sols \
    --timeout-search $timelim \
    --timeout-total  $timelim \
    --jm 13 '->' 16 : numof-clusters= $domain_size $kmax \
    sim-params= 1.0 120 \
    --jm 17 '->' 20 : numof-clusters= $domain_size $kmax \
    sim-params= 1.0 120
done
done
exit 0

--jm 1 '->' 2 : numof-clusters= $domain_size $kmax \
sim-params= 1.0 60 \
--unique-source-sinks 0 '->' 2 : voxel-side= $vside \
--jm 13 '->' 16 : numof-clusters= $domain_size $kmax \
sim-params= 1.0 60 \
--unique-source-sinks 12 '->' 16 : voxel-side= $vside \
--jm 17 '->' 20 : numof-clusters= $domain_size $kmax \
sim-params= 1.0 60 \
--unique-source-sinks 16 '->' 20 : voxel-side= $vside


#--jm 5 '->' 9  : numof-clusters= $domain_size $kmax \
#sim-params= 1.0 60 \
#--unique-source-sinks 5 '->' 9 : voxel-side= $vside
