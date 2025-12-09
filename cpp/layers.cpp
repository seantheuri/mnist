#include "layers.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdlib>

// Layer class
Layer::Layer(Layer* prev) : prev(prev) {}
Layer::~Layer() {}

Layer* Layer::get_prev() const {
    return prev;
}

std::vector<float> Layer::get_output() const {
    return output;
}

std::vector<float> Layer::get_grad_input() const {
    return grad_input;
}

void Layer::set_output(const std::vector<float>& output) {
    this->output = output;
}

void Layer::set_grad_input(const std::vector<float>& grad_input) {
    this->grad_input = grad_input;
}

// Input class
Input::Input(int size) : Layer(nullptr) {
    output.resize(size);
}

void Input::forward() {}

void Input::backward(float learning_rate) {
    (void)learning_rate;  // Unused parameter, cast to void to avoid warning
}

// Dense class
Dense::Dense(Layer* prev, int output_size) : Layer(prev) {
    int input_size = prev->get_output().size();
    weights.resize(input_size * output_size);
    bias.resize(output_size);
    grad_weights.resize(input_size * output_size);
    grad_bias.resize(output_size);

    // Initialize weights using Xavier initialization
    float scale = std::sqrt(2.0f / (input_size + output_size));
    for (float& w : weights) {
        w = static_cast<float>(rand()) / RAND_MAX * 2.0f * scale - scale;
    }
    for (float& b : bias) {
        b = 0.0f;
    }

    output.resize(output_size);
    grad_input.resize(input_size);
}

void Dense::forward() {
    const std::vector<float>& input = prev->get_output();
    int input_size = input.size();
    int output_size = output.size();

    for (int i = 0; i < output_size; ++i) {
        float sum = bias[i];
        for (int j = 0; j < input_size; ++j) {
            sum += input[j] * weights[j * output_size + i];
        }
        output[i] = sum;
    }
}

void Dense::backward(float learning_rate) {
    const std::vector<float>& input = prev->get_output();
    int input_size = input.size();
    int output_size = output.size();

    // Compute gradients for weights and biases
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            grad_weights[j * output_size + i] = input[j] * grad_input[i];
        }
        grad_bias[i] = grad_input[i];
    }

    // Compute gradients for the previous layer
    std::vector<float> prev_grad_input(input_size, 0.0f);
    for (int j = 0; j < input_size; ++j) {
        for (int i = 0; i < output_size; ++i) {
            prev_grad_input[j] += weights[j * output_size + i] * grad_input[i];
        }
    }
    prev->set_grad_input(prev_grad_input);

    // Update weights and biases
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * grad_weights[i];
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] -= learning_rate * grad_bias[i];
    }
}

// Activation class
Activation::Activation(Layer* prev, ActivationType type) : Layer(prev), type(type) {
    output.resize(prev->get_output().size());
    grad_input.resize(prev->get_output().size());
}

void Activation::forward() {
    const std::vector<float>& input = prev->get_output();
    int size = input.size();

    switch (type) {
        case ReLU:
            for (int i = 0; i < size; ++i) {
                output[i] = std::max(0.0f, input[i]);
            }
            break;
        case Tanh:
            for (int i = 0; i < size; ++i) {
                output[i] = std::tanh(input[i]);
            }
            break;
    }
}

void Activation::backward(float learning_rate) {
    const std::vector<float>& input = prev->get_output();
    int size = input.size();

    switch (type) {
        case ReLU:
            for (int i = 0; i < size; ++i) {
                grad_input[i] = (input[i] > 0) ? grad_input[i] : 0.0f;
            }
            break;
        case Tanh:
            for (int i = 0; i < size; ++i) {
                float tanh_value = std::tanh(input[i]);
                grad_input[i] *= (1 - tanh_value * tanh_value);
            }
            break;
    }

    prev->set_grad_input(grad_input);
    (void)learning_rate;  // Unused parameter, cast to void to avoid warning
}

// SoftmaxCrossEntropy class
SoftmaxCrossEntropy::SoftmaxCrossEntropy(Layer* prev) : Layer(prev) {
    output.resize(prev->get_output().size());
    grad_input.resize(prev->get_output().size());
}

void SoftmaxCrossEntropy::forward() {
    const std::vector<float>& input = prev->get_output();
    int size = input.size();

    // Compute softmax
    float max_val = *std::max_element(input.begin(), input.end());
    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] /= sum_exp;
    }
}

void SoftmaxCrossEntropy::backward(float learning_rate) {
    int size = output.size();
    for (int i = 0; i < size; ++i) {
        grad_input[i] = output[i] - grad_input[i];
    }
    prev->set_grad_input(grad_input);
}

float SoftmaxCrossEntropy::get_loss() const {
    const std::vector<float>& target = grad_input;
    int size = output.size();
    float loss = 0.0f;

    for (int i = 0; i < size; ++i) {
        loss -= target[i] * std::log(output[i] + 1e-7f);
    }

    return loss;
}

float SoftmaxCrossEntropy::get_accuracy() const {
    const std::vector<float>& target = grad_input;
    int size = output.size();
    int predicted_class = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    int true_class = std::distance(target.begin(), std::max_element(target.begin(), target.end()));

    return predicted_class == true_class ? 1.0f : 0.0f;
}
