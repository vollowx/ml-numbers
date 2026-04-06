#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MATRIX_IMPLEMENTATION
#define NEURON_IMPLEMENTATION
#include "helpers.h"
#include "matrix.h"
#include "neuron.h"

#define NUM_CLASSES 10
#define ROWS 8
#define COLS 8

typedef struct {
  int label;
  Matrix input;
} Number;

typedef struct {
  Number *items;
  size_t capacity;
  size_t count;
} Numbers;

void print_dataset(Numbers dataset) {
  size_t counts[NUM_CLASSES] = {0};
  da_foreach(Number, num, &dataset) { ++counts[num->label]; }
  for (size_t i = 0; i < NUM_CLASSES; ++i) {
    printf("'%ld' = %ld\n", i, counts[i]);
  }
}

void read_grid(FILE *fp, Matrix matrix) {
  int r = 0, c = 0;
  while (r < ROWS) {
    char ch = fgetc(fp);
    if (ch == EOF)
      break;
    if (ch == '.' || ch == 'X') {
      // Fill the matrix data array linearly: index = row * cols + col
      matrix.data[r * COLS + c] = (ch == 'X') ? 1.0f : 0.0f;
      c++;
      if (c == COLS) {
        c = 0;
        r++;
      }
    }
  }
}

int main(int argc, char **argv) {
  Numbers dataset = {0};
  Matrix expected_output[10];
  for (int num = 0; num < 10; ++num) {
    expected_output[num] = init_matrix(1, 10);
    expected_output[num].data[num] = 1;
  }

  size_t arch[] = {64, 16, 16, 10};
  Nnet nn = init_nnet(arch, sizeof(arch) / sizeof(arch[0]));
  Nnet g = init_nnet(arch, sizeof(arch) / sizeof(arch[0]));

  FILE *fp = fopen("instructions2.txt", "r");
  if (!fp) {
    printf("err: could not open instructions.txt\n");
    return 1;
  }

  char cmd[50], dummy[10];

  // TASK(20260405-214510): Use Lisp to instruct this program
  while (fscanf(fp, "%s", cmd) != EOF) {
    if (strcmp(cmd, "append") == 0) {
      Number number = {0};
      number.input = init_matrix(1, 64);
      int label;
      read_grid(fp, number.input);
      fscanf(fp, "%s %d", dummy, &number.label);

      da_append(&dataset, number);
    } else if (strcmp(cmd, "test") == 0) {
      Matrix test_input = init_matrix(1, 64);
      int expect;
      read_grid(fp, test_input);
      fscanf(fp, "%s %d", dummy, &expect);

      Matrix output = nnet_forward(nn, test_input);

      int prediction = -1;
      for (int n = 1; n < output.cols; n++)
        if (output.data[n] > output.data[prediction])
          prediction = n;

      printf("test expect %d, prediction is %d", expect, prediction);
      if (prediction != expect)
        printf("...Failing.");
      printf("\n");

      printf("Output = [ ");
      for (int i = 0; i < output.cols; ++i)
        printf("%f ", output.data[i]);
      printf("]\n");

      free_matrix(test_input);
    } else if (strcmp(cmd, "nn.pretrain") == 0) {
      int epoches;
      float lr;
      fscanf(fp, "%d %f", &epoches, &lr);

      printf("pretraining started for %d epoches\n", epoches);

      float cost;
      for (int epoch = 0; epoch < epoches; ++epoch) {
        cost = 0;
        for (int num = 0; num < 10; ++num) {
          for (int sample = 0; sample < 3; ++sample) {
            nnet_gradient(g, nn, dataset.items[num * 3 + sample].input,
                          expected_output[num], lr);
            nnet_add_inplace(nn, g);
          }
        }

        if (epoch != 0)
          printf("\033[1F");
        printf("pretraining in progress for %d epoches, epoch %d\n", epoches,
               epoch);
      }

      for (int num = 0; num < 10; ++num) {
        for (int sample = 0; sample < 3; ++sample) {
          cost += nnet_cost(nn, dataset.items[num * 3 + sample].input,
                            expected_output[num]);
        }
      }

      cost /= 30;

      printf("\033[1F");
      printf("pretraining completed for %d epoches, final cost: %f\n", epoches,
             cost);
    } else if (strcmp(cmd, "---") == 0) {
      printf("---\n");
    } else if (strcmp(cmd, "rand") == 0) {
      unsigned int seed;
      fscanf(fp, "%d", &seed);
      srand(seed == 0 ? time(0) : seed);
    } else if (strcmp(cmd, "nn.randomize") == 0) {
      nnet_randomize(nn);
    } else if (strcmp(cmd, "nn.print") == 0) {
      nnet_print(nn);
    } else if (strcmp(cmd, "print_dataset") == 0) {
      print_dataset(dataset);
    }
  }

  fclose(fp);

  da_foreach(Number, num, &dataset) { free_matrix(num->input); }
  da_free(dataset);

  free_nnet(nn);
  free_nnet(g);

  return 0;
}
