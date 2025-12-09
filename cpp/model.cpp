#include "model.hpp"
#include <iostream>
#include <cmath>

Model::Model(float learning_rate) : m_learning_rate(learning_rate) {}

Model::~Model() {
    for (Layer* layer : m_layers) {
        delete layer;
    }
}

void Model::add_layer(Layer* layer) {
    m_layers.push_back(layer);
}

void Model::train(const std::vector<float>& input_data, const std::vector<float>& target_data, int num_epochs) {
    int num_samples = input_data.size() / m_layers.front()->get_output().size();
    int batch_size = 64;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        int total_correct = 0;

        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int batch_end = std::min(batch_start + batch_size, num_samples);

            for (int i = batch_start; i < batch_end; ++i) {
                std::vector<float> input(input_data.begin() + i * m_layers.front()->get_output().size(),
                                         input_data.begin() + (i + 1) * m_layers.front()->get_output().size());
                m_layers.front()->set_output(input);

                for (Layer* layer : m_layers) {
                    layer->forward();
                }

                SoftmaxCrossEntropy* loss_layer = dynamic_cast<SoftmaxCrossEntropy*>(m_layers.back());
                std::vector<float> target(target_data.begin() + i * loss_layer->get_output().size(),
                                          target_data.begin() + (i + 1) * loss_layer->get_output().size());
                loss_layer->set_grad_input(target);

                total_loss += loss_layer->get_loss();
                total_correct += loss_layer->get_accuracy();

                for (int j = m_layers.size() - 1; j >= 0; --j) {
                    m_layers[j]->backward(m_learning_rate);
                }
            }
        }

        float avg_loss = total_loss / num_samples;
        float avg_accuracy = static_cast<float>(total_correct) / num_samples;
        std::cout << "Epoch " << epoch + 1 << " - Loss: " << avg_loss << ", Accuracy: " << avg_accuracy << std::endl;
    }
}

std::vector<float> Model::predict(const std::vector<float>& input_data) {
    m_layers.front()->set_output(input_data);

    for (Layer* layer : m_layers) {
        layer->forward();
    }

    return m_layers.back()->get_output();
}

Layer* Model::get_last_layer() const {
    if (!m_layers.empty()) {
        return m_layers.back();
    }
    return nullptr;
}
