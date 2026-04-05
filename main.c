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
  srand(time(0));
  //               I   Layer 1 (also the size of output)
  size_t arch[] = {64, 10};
  Neuron_network nn = init_neuron_network(arch, sizeof(arch) / sizeof(arch[0]));

  Numbers dataset = {0};

  FILE *fp = fopen("instructions2.txt", "r");
  if (!fp) {
    printf("err: could not open instructions.txt\n");
    return 1;
  }

  char cmd[50];
  char dummy[10];

  // TASK(20260405-214510): Use Lisp to instruct this program
  while (fscanf(fp, "%s", cmd) != EOF) {
    if (strcmp(cmd, "append") == 0) {
      Number number = {0};
      number.input = init_matrix(1, 64);
      read_grid(fp, number.input);

      // as <number>
      int label;
      fscanf(fp, "%s %d", dummy, &label);

      number.label = label;

      da_append(&dataset, number);
    } else if (strcmp(cmd, "test") == 0) {
      Matrix test_input = init_matrix(1, 64);
      read_grid(fp, test_input);

      int expect;
      fscanf(fp, "%s %d", dummy, &expect);

      Matrix output = neuron_network_forward(nn, test_input);

      int prediction = -1;
      for (int n = 1; n < NUM_CLASSES; n++)
        if (output.data[n] > output.data[prediction])
          prediction = n;

      if (prediction == expect)
        printf("test: %d (conf: %.2f%%) ... passing\n", prediction,
               output.data[prediction] * 100);
      else
        printf("test: %d (conf: %.2f%%) ... failing, expected: %d\n",
               prediction, output.data[prediction] * 100, expect);

      free_matrix(test_input);
      free_matrix(output);
    } else if (strcmp(cmd, "nn_train") == 0) {
      int epoches;
      float lr;
      fscanf(fp, "%d %f", &epoches, &lr);

      printf("training started for %d epoches\n", epoches);

      Matrix expected_output[10];
      for (int num = 0; num < 10; ++num) {
        expected_output[num] = init_matrix(1, 10);
        expected_output[num].data[num] = 1;
      }

      float loss;
      for (int epoch = 0; epoch < epoches; ++epoch) {
        loss = 0;
        for (int num = 0; num < 10; ++num) {
          for (int sample = 0; sample < 3; ++sample) {
            neuron_network_train(&nn, dataset.items[num * 3 + sample].input,
                                 expected_output[num], lr);
            loss +=
                neuron_network_loss(nn, dataset.items[num * 3 + sample].input,
                                    expected_output[num]);
          }
        }
        printf("\033[F");
        printf("training, epoch %5d, loss: %f\n", epoch, loss);
      }

      printf("\033[F");
      printf("training completed for %d epoches, final loss: %f\033[0K\n",
             epoches, loss);
    } else if (strcmp(cmd, "print_divider") == 0) {
      printf("---\n");
    } else if (strcmp(cmd, "nn_randomize") == 0) {
      neuron_network_randomize(nn);
    } else if (strcmp(cmd, "nn_print") == 0) {
      neuron_network_print(nn);
    } else if (strcmp(cmd, "print_dataset") == 0) {
      print_dataset(dataset);
    }
  }

  fclose(fp);

  da_foreach(Number, num, &dataset) { free_matrix(num->input); }
  da_free(dataset);

  free_neuron_network(nn);

  return 0;
}
