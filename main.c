#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helpers.h"

#define NUM_CLASSES 10
#define ROWS 8
#define COLS 8

typedef struct {
  int number;
  int look[ROWS][COLS];
} Number;

typedef struct {
  Number *items;
  size_t capacity;
  size_t count;
} Numbers;

typedef struct {
  float weights[ROWS][COLS];
  float bias;
} Model;

Numbers dataset = {0};
Model models[NUM_CLASSES] = {0};

void print_model(Model m) {
  printf("model = {\n"
         "  bias = %f,\n"
         "  weights = \n",
         m.bias);

  // for (size_t i = 0; i < ROWS; ++i) {
  //   printf("    ");
  //   for (size_t j = 0; j < COLS; ++j) {
  //     printf("%f ", m.weights[i][j]);
  //   }
  //   printf("\n");
  // }
  // printf("  }\n"
  //        "}\n");

  float max_abs = 0.01f; // 防止除以零的微小初值
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      float abs_val = fabsf(m.weights[i][j]);
      if (abs_val > max_abs)
        max_abs = abs_val;
    }
  }

  for (size_t i = 0; i < ROWS; ++i) {
    printf("    ");
    for (size_t j = 0; j < COLS; ++j) {
      float w = m.weights[i][j];
      int r = 0, g = 0, b = 0;

      float intensity = fabsf(w) / max_abs;
      int color_val = (int)(intensity * 255);

      if (w > 0)
        g = color_val;
      else if (w < 0)
        r = color_val;

      printf("\033[48;2;%d;%d;%dm  \033[0m", r, g, b);
    }
    printf("\n");
  }
  printf("  }\n"
         "}\n");
}

void print_dataset() {
  size_t counts[NUM_CLASSES] = {0};
  da_foreach(Number, num, &dataset) { ++counts[num->number]; }
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

void forward(int input[ROWS][COLS], float *output) {
  for (int n = 0; n < NUM_CLASSES; n++) {
    float sum = models[n].bias;
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        sum += models[n].weights[r][c] * input[r][c];
      }
    }
    output[n] = sum;
  }
  softmax(output, NUM_CLASSES);
}

void train(int cycles, float lr) {
  if (dataset.count == 0) {
    printf("err: no data to train on.\n");
    return;
  }
  printf("training started for %d cycle\n", cycles);

  float total_loss = 0;

  for (int cycle = 1; cycle <= cycles; cycle++) {
    total_loss = 0;

    da_foreach(Number, num, &dataset) {
      float probs[NUM_CLASSES];
      forward(num->look, probs);
      for (int n = 0; n < NUM_CLASSES; n++) {
        float target = (num->number == n) ? 1.0f : 0.0f;
        float error = target - probs[n];
        total_loss += error * error;
        for (int r = 0; r < ROWS; ++r) {
          for (int c = 0; c < COLS; ++c) {
            models[n].weights[r][c] += lr * error * num->look[r][c];
          }
        }
        models[n].bias += lr * error;
      }
    }

    printf("\033[F");
    printf("training, cycle %5d, loss: %f\n", cycle,
           total_loss / dataset.count);
  }
  printf("\033[F");
  printf("training completed for %d cycle, loss: %f\033[0K\n", cycles,
         total_loss);
}

void read_grid(FILE *fp, int grid[ROWS][COLS]) {
  int r = 0, c = 0;
  while (r < ROWS) {
    char ch = fgetc(fp);
    if (ch == EOF)
      break;
    if (ch == '.' || ch == 'X') {
      grid[r][c] = (ch == 'X') ? 1 : 0;
      c++;
      if (c == COLS) {
        c = 0;
        r++;
      }
    }
  }
}

int main() {
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
      read_grid(fp, number.look);

      // as <number>
      int label;
      fscanf(fp, "%s %d", dummy, &label);

      number.number = label;

      da_append(&dataset, number);
    } else if (strcmp(cmd, "test") == 0) {
      int test_grid[ROWS][COLS];
      read_grid(fp, test_grid);

      // expect <number>
      int expect;
      fscanf(fp, "%s %d", dummy, &expect);

      float probs[NUM_CLASSES];
      forward(test_grid, probs);

      int prediction = 0;
      for (int n = 1; n < NUM_CLASSES; n++) {
        if (probs[n] > probs[prediction])
          prediction = n;
      }

      if (prediction == expect)
        printf("test: %d (conf: %.2f%%) ... passing\n", prediction,
               probs[prediction] * 100);
      else
        printf("test: %d (conf: %.2f%%) ... failing, expected: %d\n",
               prediction, probs[prediction] * 100, expect);
    } else if (strcmp(cmd, "train") == 0) {
      int cycles;
      fscanf(fp, "%d", &cycles);
      train(cycles, 0.1f);
    } else if (strcmp(cmd, "print_divider") == 0) {
      printf("---\n");
    } else if (strcmp(cmd, "print_model") == 0) {
      size_t model_id;
      fscanf(fp, "%ld", &model_id);

      if (model_id > 9)
        continue;

      print_model(models[model_id]);
    } else if (strcmp(cmd, "print_dataset") == 0) {
      print_dataset();
    }
  }

  fclose(fp);
  return 0;
}
