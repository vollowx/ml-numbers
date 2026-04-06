#include <stdio.h>
#include <time.h>

#define MATRIX_IMPLEMENTATION
#include "matrix.h"
#define NEURON_IMPLEMENTATION
#include "neuron.h"

int main(void) {
  srand(time(0));

  // This defines:
  // - Input: 2
  // - Layer 1: 2 neurons
  // - Layer 2: 1 neuron
  size_t arch[] = {2, 2, 1};
  Nnet nn = init_nnet(arch, sizeof(arch) / sizeof(arch[0]));
  nnet_randomize(nn);

  Matrix inputs[4];
  Matrix targets[4];
  float data[4][2] = {
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1},
  };
  float goals[4][1] = {
      {0},
      {1},
      {1},
      {0},
  };

  for (int i = 0; i < 4; ++i) {
    inputs[i] = init_matrix(1, 2);
    targets[i] = init_matrix(1, 1);
    inputs[i].data[0] = data[i][0];
    inputs[i].data[1] = data[i][1];
    targets[i].data[0] = goals[i][0];
  }

  int epochs = 20000;
  float lr = 1e-3f;

  Matrix *errors = calloc(nn.count, sizeof(Matrix));
  for (size_t i = 0; i < nn.count; i++) {
    errors[i] = init_matrix(1, nn.layers[i].a.cols);
  }

  for (int e = 0; e < epochs; ++e) {
    int i = rand() % 4; // Stochastic gradient descent
    nnet_train(nn, inputs[i], targets[i], lr, errors);

    if (e % 5000 == 0) {
      float loss = nnet_loss(nn, inputs[i], targets[i]);
      printf("Epoch %d, Current Sample Loss: %f\n", e, loss);
    }
  }

  printf("\nResults:\n");
  for (int i = 0; i < 4; ++i) {
    Matrix out = nnet_forward(nn, inputs[i]);
    printf("[%.0f, %.0f] -> %f (expected: %.0f)\n", inputs[i].data[0],
           inputs[i].data[1], out.data[0], targets[i].data[0]);
  }

  nnet_print(nn);

  // 5. Cleanup
  for (int i = 0; i < 4; ++i) {
    free_matrix(inputs[i]);
    free_matrix(targets[i]);
  }
  free_nnet(nn);

  return 0;

  return 0;
}
