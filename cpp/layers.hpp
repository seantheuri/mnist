#pragma once

#include <vector>

class Layer {
public:
    Layer(Layer* prev);
    virtual ~Layer();

    virtual void forward() = 0;
    virtual void backward(float learning_rate) = 0;

    Layer* get_prev() const;

    std::vector<float> get_output() const;
    std::vector<float> get_grad_input() const;

    void set_output(const std::vector<float>& output);
    void set_grad_input(const std::vector<float>& grad_input);

protected:
    Layer* prev;
    std::vector<float> output;
    std::vector<float> grad_input;
};

class Input : public Layer {
public:
    Input(int size);
    void forward() override;
    void backward(float learning_rate) override;
};

class Dense : public Layer {
public:
    Dense(Layer* prev, int output_size);
    void forward() override;
    void backward(float learning_rate) override;

private:
    std::vector<float> weights;
    std::vector<float> bias;
    std::vector<float> grad_weights;
    std::vector<float> grad_bias;
};

class Activation : public Layer {
public:
    enum ActivationType {
        ReLU,
        Tanh
    };

    Activation(Layer* prev, ActivationType type);
    void forward() override;
    void backward(float learning_rate) override;

private:
    ActivationType type;
};

class SoftmaxCrossEntropy : public Layer {
public:
    SoftmaxCrossEntropy(Layer* prev);
    void forward() override;
    void backward(float learning_rate) override;

    float get_loss() const;
    float get_accuracy() const;

private:
    float loss;
    float accuracy;
};
