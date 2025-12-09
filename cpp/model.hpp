#pragma once

#include <vector>
#include "layers.hpp"

class Model {
public:
    Model(float learning_rate);
    ~Model();

    void add_layer(Layer* layer);
    void train(const std::vector<float>& input_data, const std::vector<float>& target_data, int num_epochs);
    std::vector<float> predict(const std::vector<float>& input_data);

    Layer* get_last_layer() const;

private:
    std::vector<Layer*> m_layers;
    float m_learning_rate;
};