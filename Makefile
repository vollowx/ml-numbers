all: main xor nomatrix

nomatrix: nomatrix.c Makefile
	gcc -lm -O3 -o nomatrix nomatrix.c

xor: xor.c matrix.h neuron.h Makefile
	gcc -lm -O3 -o xor xor.c

main: main.c matrix.h neuron.h Makefile
	gcc -lm -O3 -o main main.c
