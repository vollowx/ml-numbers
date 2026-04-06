CFLAGS = -O3 -Wall -Wextra -I.
LIBS   = -lm

all: main xor nomatrix

nomatrix: ./playground/nomatrix.c Makefile
	gcc $(CFLAGS) $(LIBS) -o ./playground/nomatrix ./playground/nomatrix.c

xor: ./playground/xor.c matrix.h neuron.h Makefile
	gcc $(CFLAGS) $(LIBS) -o ./playground/xor ./playground/xor.c

main: ./playground/main.c matrix.h neuron.h Makefile
	gcc $(CFLAGS) $(LIBS) -o ./playground/main ./playground/main.c

clean:
	rm -f ./playground/main ./playground/xor ./playground/nomatrix
