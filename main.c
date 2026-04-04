#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helpers.h"
#define MATRIX_IMPLEMENTATION
#include "matrix.h"

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

typedef struct {
  Matrix weights;
  Matrix biases;
} Model;

void print_model(Model m) {}

void print_dataset(Numbers dataset) {
  size_t counts[NUM_CLASSES] = {0};
  da_foreach(Number, num, &dataset) { ++counts[num->label]; }
  for (size_t i = 0; i < NUM_CLASSES; ++i) {
    printf("'%ld' = %ld\n", i, counts[i]);
  }
}

void softmax(float *inputs, int size) {
  float sum = 0.0, max_val = inputs[0];
  for (int i = 1; i < size; i++)
    if (inputs[i] > max_val)
      max_val = inputs[i];
  for (int i = 0; i < size; i++) {
    inputs[i] = expf(inputs[i] - max_val);
    sum += inputs[i];
  }
  for (int i = 0; i < size; i++)
    inputs[i] /= sum;
}

float loss(Matrix probs, int label) {
  float cost = 0;
  for (int i = 0; i < NUM_CLASSES; i++) {
    float target = (i == label) ? 1.0f : 0.0f;
    float error = target - probs.data[i];
    cost += error * error;
  }
  return cost;
}

void forward(Matrix out, Model *m, Matrix input) {
  matrix_mul_transposed_b(out, input, m->weights);
  matrix_plus_inplace(out, m->biases);

  softmax(out.data, NUM_CLASSES);
}

void train(Model *m, Numbers dataset, int cycles, float lr) {
  if (dataset.count == 0) {
    printf("err: no data to train on.\n");
    return;
  }
  printf("training started for %d cycles\n", cycles);

  Matrix probs = init_matrix(1, 10);
  float total_loss = 0;

  for (int cycle = 1; cycle <= cycles; cycle++) {
    float total_loss = 0;

    da_foreach(Number, num, &dataset) {
      forward(probs, m, num->input);

      // Calculate Loss for reporting
      total_loss += loss(probs, num->label);

      // Backpropagation / Weight Update
      for (int n = 0; n < NUM_CLASSES; n++) {
        float target = (num->label == n) ? 1.0f : 0.0f;
        float error = target - probs.data[n];

        // Update the weights for neuron 'n'
        // These weights are stored in the n-th column of the matrix
        for (int i = 0; i < ROWS * COLS; i++) {
          // Indexing into a 64x10 matrix: row i, column n
          m->weights.data[n * 64 + i] += lr * error * num->input.data[i];
        }

        m->biases.data[n] += lr * error;
      }
    }

    printf("\033[F");
    printf("training, cycle %5d, loss: %f\n", cycle,
           total_loss / dataset.count);
  }

  free_matrix(probs);

  printf("\033[F");
  printf("training completed for %d cycles, loss: %f\033[0K\n", cycles,
         total_loss);
}

void read_grid(FILE *fp, Matrix *matrix) {
  int r = 0, c = 0;
  while (r < ROWS) {
    char ch = fgetc(fp);
    if (ch == EOF)
      break;
    if (ch == '.' || ch == 'X') {
      // Fill the matrix data array linearly: index = row * cols + col
      matrix->data[r * COLS + c] = (ch == 'X') ? 1.0f : 0.0f;
      c++;
      if (c == COLS) {
        c = 0;
        r++;
      }
    }
  }
}

int main() {
  Numbers dataset = {0};
  Model model = {0};
  model.weights = init_matrix(10, 64); // Transposed
  model.biases = init_matrix(1, 10);

  FILE *fp = fopen("instructions.txt", "r");
  if (!fp) {
    printf("err: could not open instructions.txt\n");
    return 1;
  }

  char cmd[50];
  char dummy[10];

  while (fscanf(fp, "%s", cmd) != EOF) {
    if (strcmp(cmd, "append") == 0) {
      Number number = {0};
      number.input = init_matrix(1, 64);
      read_grid(fp, &number.input);

      // as <number>
      int label;
      fscanf(fp, "%s %d", dummy, &label);

      number.label = label;

      da_append(&dataset, number);
    } else if (strcmp(cmd, "test") == 0) {
      Matrix test_input = init_matrix(1, 64);
      read_grid(fp, &test_input);

      int expect;
      fscanf(fp, "%s %d", dummy, &expect);

      Matrix probs = init_matrix(1, 10);
      forward(probs, &model, test_input);

      int prediction = 0;
      for (int n = 1; n < NUM_CLASSES; n++) {
        if (probs.data[n] > probs.data[prediction])
          prediction = n;
      }

      if (prediction == expect)
        printf("test: %d (conf: %.2f%%) ... passing\n", prediction,
               probs.data[prediction] * 100);
      else
        printf("test: %d (conf: %.2f%%) ... failing, expected: %d\n",
               prediction, probs.data[prediction] * 100, expect);

      free_matrix(test_input);
      free_matrix(probs);
    } else if (strcmp(cmd, "train") == 0) {
      int cycles;
      fscanf(fp, "%d", &cycles);
      train(&model, dataset, cycles, 0.1f);
    } else if (strcmp(cmd, "print_divider") == 0) {
      printf("---\n");
    } else if (strcmp(cmd, "print_model") == 0) {
      size_t model_id;
      fscanf(fp, "%ld", &model_id);

      if (model_id > 9)
        continue;

      // print_model(neurons[model_id]);
    } else if (strcmp(cmd, "print_dataset") == 0) {
      print_dataset(dataset);
    }
  }

  fclose(fp);

  da_foreach(Number, num, &dataset) { free_matrix(num->input); }
  da_free(dataset);

  free_matrix(model.weights);
  free_matrix(model.biases);

  return 0;
}
