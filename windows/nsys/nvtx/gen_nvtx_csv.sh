#!/bin/bash
# nsys stats --report=cuda_gpu_trace,nvtx_pushpop_trace --format csv --output=node0 traces/0_41047697.nsys-rep
# nsys stats --report=cuda_gpu_trace,nvtx_pushpop_trace --format csv --output=node2 traces/2_41047697.nsys-rep
nsys stats --report=cuda_gpu_trace,nvtx_pushpop_trace --format csv --output=node1 traces/1_41047697.nsys-rep
nsys stats --report=cuda_gpu_trace,nvtx_pushpop_trace --format csv --output=node3 traces/3_41047697.nsys-rep
