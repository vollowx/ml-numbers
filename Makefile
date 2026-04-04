main: main.c number_recognition.c Makefile
	gcc -lm -o main main.c
	gcc -lm -o numberr number_recognition.c
