all: oneesan

oneesan: main.cu
	nvcc -o $@ $^ -O3 --generate-code arch=compute_60,code=sm_86 -Xcompiler -fopenmp

.PHONY: clean
clean:
	-rm oneesan
