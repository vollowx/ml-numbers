#ifndef NEURON_H
#define NEURON_H

#include "matrix.h"

float sigmoid(double x);
float sigmoidf(float x);
// Derivative of sigmoid, x is the result of `sigmoid(n)`
float dsigmoid(double x);
float dsigmoidf(float x);

void softmax(float *inputs, int size);

typedef struct {
  Matrix weights;
  Matrix biases;
  Matrix output;
} Layer;

typedef struct {
  Layer *layers;
  size_t count;
} Neuron_network;

Neuron_network init_neuron_network(size_t *arch, size_t n_arch);
void free_neuron_network(Neuron_network nn);
void neuron_network_randomize(Neuron_network nn);
Matrix neuron_network_forward(Neuron_network nn, Matrix input);
void neuron_network_train(Neuron_network *nn, Matrix input, Matrix expectation,
                          float lr);
float neuron_network_loss(Neuron_network nn, Matrix input, Matrix expectation);
// Derivative of the loss of a neuron network
float neuron_network_dloss(Neuron_network nn, Matrix input, Matrix expectation);
void neuron_network_print(Neuron_network nn);

#endif // NEURON_H_

#ifdef NEURON_IMPLEMENTATION
#ifndef NEURON_IMPLEMENTATION_ONCE
#define NEURON_IMPLEMENTATION_ONCE

#include <math.h>

float sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
float dsigmoid(double x) { return x * (1.0 - x); }
float dsigmoidf(float x) { return x * (1.0f - x); }

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

Neuron_network init_neuron_network(size_t *arch, size_t n_arch) {
  Neuron_network nn = {0};

  nn.count = n_arch - 1;
  nn.layers = calloc(nn.count, sizeof(Layer));

  for (size_t i = 0; i < nn.count; ++i) {
    size_t n_input = arch[i];
    size_t n_neuron = arch[i + 1];

    nn.layers[i].weights = init_matrix(n_input, n_neuron);
    nn.layers[i].biases = init_matrix(1, n_neuron);
    nn.layers[i].output = init_matrix(1, n_neuron);
  }

  return nn;
}

// TASK(20260405-214436): Implement free_neuron_network
void free_neuron_network(Neuron_network nn) {};

void neuron_network_randomize(Neuron_network nn) {
  for (size_t i = 0; i < nn.count; ++i) {
    for (size_t j = 0;
         j < nn.layers[i].weights.rows * nn.layers[i].weights.cols; ++j) {
      // (-0.5, 0.5)
      nn.layers[i].weights.data[j] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    }
  }
}

Matrix neuron_network_forward(Neuron_network nn, Matrix input) {
  Matrix current_input = input;

  for (size_t i = 0; i < nn.count; ++i) {
    Layer *l = &nn.layers[i];
    matrix_mul(l->output, current_input, l->weights);
    matrix_add_inplace(l->output, l->biases);

    for (size_t j = 0; j < l->output.rows * l->output.cols; ++j) {
      l->output.data[j] = sigmoidf(l->output.data[j]);
    }

    current_input = l->output;
  }

  return nn.layers[nn.count - 1].output;
}

void neuron_network_train(Neuron_network *nn, Matrix input, Matrix expectation,
                          float lr) {
  neuron_network_forward(*nn, input);

  Matrix *errors = calloc(nn->count, sizeof(Matrix));
  for (size_t i = 0; i < nn->count; i++) {
    errors[i] = init_matrix(1, nn->layers[i].output.cols);
  }

  Layer *output_layer = &nn->layers[nn->count - 1];
  for (size_t i = 0; i < output_layer->output.cols; i++) {
    // Error = (Expectation - Output)
    errors[nn->count - 1].data[i] =
        expectation.data[i] - output_layer->output.data[i];
  }

  // Backpropagation
  for (int i = nn->count - 1; i >= 0; i--) {
    Layer *l = &nn->layers[i];
    Matrix current_input = (i == 0) ? input : nn->layers[i - 1].output;
    Matrix current_error = errors[i];

    for (size_t j = 0; j < l->weights.cols; j++) {
      float gradient =
          current_error.data[j] * dsigmoidf(l->output.data[j]) * lr;

      l->biases.data[j] += gradient;
      for (size_t k = 0; k < l->weights.rows; k++) {
        matrix_at(l->weights, k, j) += gradient * current_input.data[k];
      }
    }

    if (i > 0) {
      Layer *prev_layer = &nn->layers[i - 1];
      Matrix prev_error = errors[i - 1];

      for (size_t j = 0; j < prev_layer->output.cols; j++) {
        float err_sum = 0.0f;
        for (size_t k = 0; k < l->weights.cols; k++) {
          err_sum += current_error.data[k] * matrix_at(l->weights, j, k);
        }
        prev_error.data[j] = err_sum;
      }
    }
  }

  for (size_t i = 0; i < nn->count; i++)
    free_matrix(errors[i]);
  free(errors);
}

float neuron_network_loss(Neuron_network nn, Matrix input, Matrix expectation) {
  Matrix output = neuron_network_forward(nn, input);

  assert(output.cols == expectation.cols);
  assert(output.rows == expectation.rows);

  float loss = 0.0f;
  size_t n = output.rows * output.cols;

  for (size_t i = 0; i < n; ++i) {
    float error = expectation.data[i] - output.data[i];
    loss += error * error;
  }

  return loss / (float)n;
}

void neuron_network_print(Neuron_network nn) {
  printf("Layers: %zu\n", nn.count);

  for (size_t i = 0; i < nn.count; ++i) {
    Layer l = nn.layers[i];
    printf("Layer %zu:\n", i + 1);

    // Print Weights
    printf("  Weights (%zu x %zu):\n", l.weights.rows, l.weights.cols);
    for (size_t r = 0; r < l.weights.rows; ++r) {
      printf("    [ ");
      for (size_t c = 0; c < l.weights.cols; ++c) {
        printf("%6.3f ", matrix_at(l.weights, r, c));
      }
      printf("]\n");
    }

    // Print Biases
    printf("  Biases (%zu x %zu):\n", l.biases.rows, l.biases.cols);
    printf("    [ ");
    for (size_t c = 0; c < l.biases.cols; ++c) {
      printf("%6.3f ", l.biases.data[c]);
    }
    printf("]\n\n");
  }
}

#endif // NEURON_IMPLEMENTATION_ONCE
#endif // NEURON_IMPLEMENTATION
