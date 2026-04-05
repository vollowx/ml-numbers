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
  Neural_net nn = init_neural_net(arch, sizeof(arch) / sizeof(arch[0]));
  neural_net_randomize(nn);

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

  for (int e = 0; e < epochs; ++e) {
    int i = rand() % 4; // Stochastic gradient descent
    neural_net_train(nn, inputs[i], targets[i], lr);

    if (e % 5000 == 0) {
      float loss = neural_net_loss(nn, inputs[i], targets[i]);
      printf("Epoch %d, Current Sample Loss: %f\n", e, loss);
    }
  }

  printf("\nResults:\n");
  for (int i = 0; i < 4; ++i) {
    Matrix out = neural_net_forward(nn, inputs[i]);
    printf("[%.0f, %.0f] -> %f (expected: %.0f)\n", inputs[i].data[0],
           inputs[i].data[1], out.data[0], targets[i].data[0]);
  }

  neural_net_print(nn);

  // 5. Cleanup
  for (int i = 0; i < 4; ++i) {
    free_matrix(inputs[i]);
    free_matrix(targets[i]);
  }
  free_neural_net(nn);

  return 0;

  return 0;
}
