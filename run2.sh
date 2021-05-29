#!/bin/bash

rm -f log
for ((M=128; M<=1024; M=M*2))
do
	for ((K=16; K<=1024; K=K*2))
	do
		cd ../data
		./gen_data $M $M $K
		cd - > /dev/null
		./gemm 4 >> log
	done

	for ((K=16; K<=1024; K=K*2))
	do
		./gemm 8 >> log
	done
	
	for ((K=16; K<=1024; K=K*2))
	do
		./gemm 16 >> log
	done

	for ((K=16; K<=1024; K=K*2))
	do
		./gemm 32 >> log
	done	

	for ((K=16; K<=1024; K=K*2))
	do
		./gemm 64 >> log
	done
	
	for ((K=16; K<=1024; K=K*2))
	do
		./gemm 128 >> log
	done

	for ((K=16; K<=1024; K=K*2))
	do
		./gemm 256 >> log00
	done
done
