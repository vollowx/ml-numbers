main: main.c number_recognition.c Makefile
	gcc -lm -O3 -o main main.c
	gcc -lm -O3 -o numberr number_recognition.c
