#ifndef NEURON_H
#define NEURON_H

#include "matrix.h"

#define output_layer(nn) nn.layers[nn.n_layer - 1]

float sigmoid(double x);
float sigmoidf(float x);
// Derivative of sigmoid, x is the result of `sigmoid(n)`
float dsigmoid(double x);
float dsigmoidf(float x);

void softmax(float *inputs, int size);

typedef struct {
  Matrix w;
  Matrix b;
  Matrix a;
} Layer;

typedef struct {
  Layer *layers;
  size_t n_layer;
} Nnet;

// Allocation included
Nnet init_nnet(size_t *arch, size_t n_arch);
void free_nnet(Nnet nn);
void nnet_randomize(Nnet nn);
Matrix nnet_forward(Nnet nn, Matrix input);
void nnet_gradient(Nnet g, Nnet nn, Matrix input, Matrix expectation, float lr);
void nnet_add(Nnet out, Nnet a, Nnet b);
void nnet_add_inplace(Nnet out, Nnet b);
float nnet_cost(Nnet nn, Matrix input, Matrix expectation);
void nnet_print(Nnet nn);

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

Nnet init_nnet(size_t *arch, size_t n_arch) {
  Nnet nn = {0};

  nn.n_layer = n_arch - 1;
  nn.layers = calloc(nn.n_layer, sizeof(Layer));

  for (size_t i = 0; i < nn.n_layer; ++i) {
    size_t n_input = arch[i];
    size_t n_neuron = arch[i + 1];

    nn.layers[i].w = init_matrix(n_input, n_neuron);
    nn.layers[i].b = init_matrix(1, n_neuron);
    nn.layers[i].a = init_matrix(1, n_neuron);
  }

  return nn;
}

void free_nnet(Nnet nn) {
  for (size_t i = 0; i < nn.n_layer; ++i) {
    free_matrix(nn.layers[i].w);
    free_matrix(nn.layers[i].b);
    free_matrix(nn.layers[i].a);
  }
  free(nn.layers);
};

void nnet_randomize(Nnet nn) {
  for (size_t i = 0; i < nn.n_layer; ++i) {
    for (size_t j = 0; j < nn.layers[i].w.rows * nn.layers[i].w.cols; ++j) {
      // (-0.5, 0.5)
      nn.layers[i].w.data[j] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    }
  }
}

Matrix nnet_forward(Nnet nn, Matrix input) {
  Matrix crt_input = input;

  for (size_t i = 0; i < nn.n_layer; ++i) {
    Layer *l = &nn.layers[i];
    matrix_mul(l->a, crt_input, l->w);
    matrix_add_inplace(l->a, l->b);

    for (size_t j = 0; j < l->a.rows * l->a.cols; ++j) {
      l->a.data[j] = sigmoidf(l->a.data[j]);
    }

    crt_input = l->a;
  }

  return nn.layers[nn.n_layer - 1].a;
}

void nnet_gradient(Nnet g, Nnet nn, Matrix input, Matrix expectation,
                   float lr) {
  // The `a` of layers of `g` is used as the error container

  nnet_forward(nn, input);
  for (int i = nn.n_layer - 1; i >= 0; i--) {
    Layer *crt_layer_g = &g.layers[i];
    Layer *crt_layer = &nn.layers[i];
    Matrix crt_input = (i == 0) ? input : nn.layers[i - 1].a;

    for (size_t j = 0; j < crt_layer->w.cols; j++) {
      if (i == (int)nn.n_layer - 1)
        crt_layer_g->a.data[j] = expectation.data[j] - crt_layer->a.data[j];

      float gradient =
          crt_layer_g->a.data[j] * dsigmoidf(crt_layer->a.data[j]) * lr;

      for (size_t k = 0; k < crt_layer->w.rows; ++k)
        matrix_at(crt_layer_g->w, k, j) = gradient * crt_input.data[k];
      crt_layer_g->b.data[j] = gradient;
    }

    // Backpropagation
    if (i > 0)
      // Not really a transposed GEMM, but it just fits such situation
      matrix_mul_transposed_b(g.layers[i - 1].a, crt_layer_g->a, crt_layer->w);
  }
}

void nnet_add_inplace(Nnet out, Nnet b) {
  assert(out.n_layer == b.n_layer);
  // Assume that the architectures equal

  for (size_t i = 0; i < out.n_layer; i++) {
    Layer *l_out = &out.layers[i];
    Layer *l_b = &b.layers[i];

    for (size_t j = 0; j < l_out->w.rows * l_out->w.cols; j++)
      l_out->w.data[j] += l_b->w.data[j];

    for (size_t j = 0; j < l_out->b.rows * l_out->b.cols; j++)
      l_out->b.data[j] += l_b->b.data[j];
  }
}

float nnet_cost(Nnet nn, Matrix input, Matrix expectation) {
  Matrix output = nnet_forward(nn, input);

  assert(output.cols == expectation.cols);
  assert(output.rows == expectation.rows);

  float cost = 0.0f;
  size_t n = output.rows * output.cols;

  for (size_t i = 0; i < n; ++i) {
    float error = expectation.data[i] - output.data[i];
    cost += error * error;
  }

  return cost / (float)n;
}

void nnet_print(Nnet nn) {
  printf("Layers: %zu\n", nn.n_layer);

  for (size_t i = 0; i < nn.n_layer; ++i) {
    Layer l = nn.layers[i];
    printf("Layer %zu:\n", i + 1);

    // Print Weights
    printf("  Weights (%zu x %zu):\n", l.w.rows, l.w.cols);
    for (size_t r = 0; r < l.w.rows; ++r) {
      printf("    [ ");
      for (size_t c = 0; c < l.w.cols; ++c) {
        printf("%6.3f ", matrix_at(l.w, r, c));
      }
      printf("]\n");
    }

    // Print Biases
    printf("  Biases (%zu x %zu):\n", l.b.rows, l.b.cols);
    printf("    [ ");
    for (size_t c = 0; c < l.b.cols; ++c) {
      printf("%6.3f ", l.b.data[c]);
    }
    printf("]\n\n");
  }
}

#endif // NEURON_IMPLEMENTATION_ONCE
#endif // NEURON_IMPLEMENTATION
