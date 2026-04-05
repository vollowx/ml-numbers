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
  size_t arch[] = {64, 10};
  Neural_net nn = init_neural_net(arch, sizeof(arch) / sizeof(arch[0]));

  Numbers dataset = {0};
  Matrix expected_output[10];
  for (int num = 0; num < 10; ++num) {
    expected_output[num] = init_matrix(1, 10);
    expected_output[num].data[num] = 1;
  }

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

      Matrix output = neural_net_forward(nn, test_input);

      int prediction = -1;
      for (int n = 1; n < output.cols; n++)
        if (output.data[n] > output.data[prediction])
          prediction = n;

      if (prediction == expect)
        printf("test: %d from %f ... √\n", prediction, output.data[prediction]);
      else
        printf("test: %d from %f ... ×, expected: %d\n", prediction,
               output.data[prediction], expect);

      printf("\t\t\t\t\t[ ");
      for (int i = 0; i < output.cols; ++i)
        printf("\033[%dm%f\033[0m ", output.data[i] > 0.5 ? 32 : 90,
               output.data[i]);
      printf("]\n");

      free_matrix(test_input);
      free_matrix(output);
    } else if (strcmp(cmd, "nn_train") == 0) {
      int epoches;
      float lr;
      fscanf(fp, "%d %f", &epoches, &lr);

      printf("training started for %d epoches\n", epoches);

      float loss;
      for (int epoch = 0; epoch < epoches; ++epoch) {
        loss = 0;
        for (int num = 0; num < 10; ++num) {
          for (int sample = 0; sample < 3; ++sample) {
            neural_net_train(nn, dataset.items[num * 3 + sample].input,
                             expected_output[num], lr);
            loss += neural_net_loss(nn, dataset.items[num * 3 + sample].input,
                                    expected_output[num]);
          }
        }
        printf("\033[F");
        printf("training, epoch %5d, loss: %f\n", epoch, loss);
      }

      printf("\033[F");
      printf("training completed for %d epoches, final loss: %f\033[0K\n",
             epoches, loss);
    } else if (strcmp(cmd, "---") == 0) {
      printf("---\n");
    } else if (strcmp(cmd, "rand") == 0) {
      unsigned int seed;
      fscanf(fp, "%d", &seed);
      srand(seed == 0 ? time(0) : seed);
    } else if (strcmp(cmd, "nn_randomize") == 0) {
      neural_net_randomize(nn);
    } else if (strcmp(cmd, "nn_print") == 0) {
      neural_net_print(nn);
    } else if (strcmp(cmd, "print_dataset") == 0) {
      print_dataset(dataset);
    }
  }

  fclose(fp);

  da_foreach(Number, num, &dataset) { free_matrix(num->input); }
  da_free(dataset);

  free_neural_net(nn);

  return 0;
}
