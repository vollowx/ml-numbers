all: main nomatrix

main: main.c matrix.h Makefile
	gcc -lm -O3 -o main main.c

nomatrix: nomatrix.c Makefile
	gcc -lm -O3 -o nomatrix nomatrix.c
