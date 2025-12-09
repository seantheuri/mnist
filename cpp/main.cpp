#include "model.hpp"
#include "layers.hpp"
#include "MNISTParser.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

int main(int argc, char** argv) {
    // Directory containing the Fashion MNIST dataset
    std::string data_directory = "/home/stheuri/lab5/dataset";

    // Parse command line arguments
    if (argc > 1) {
        data_directory = argv[1];
    }

    // Load Fashion MNIST training dataset
    std::string train_image_file = data_directory + "/train-images.idx3-ubyte";
    std::string train_label_file = data_directory + "/train-labels.idx1-ubyte";
    int num_train_images, num_channels, height, width, num_classes;
    float* train_image_data;
    float* train_label_data;

    LoadMNISTData(train_image_file, train_label_file, num_train_images, num_channels, height, width, num_classes, &train_image_data, &train_label_data);

    // Create a model
    float learning_rate = 0.001f;
    Model model(learning_rate);

    // Add layers to the model
    model.add_layer(new Input(height * width));
    model.add_layer(new Dense(model.get_last_layer(), 128));
    model.add_layer(new Activation(model.get_last_layer(), Activation::ReLU));
    model.add_layer(new Dense(model.get_last_layer(), num_classes));
    model.add_layer(new SoftmaxCrossEntropy(model.get_last_layer()));

    // Prepare input data and target data
    std::vector<float> input_data(train_image_data, train_image_data + num_train_images * height * width);
    std::vector<float> target_data(train_label_data, train_label_data + num_train_images * num_classes);

    // Train the model
    int num_epochs = 10;
    model.train(input_data, target_data, num_epochs);

    // Free the training dataset
    delete[] train_image_data;
    delete[] train_label_data;

    // Load Fashion MNIST test dataset
    std::string test_image_file = data_directory + "test-images.idx3-ubyte";
    std::string test_label_file = data_directory + "test-labels.idx1-ubyte";
    int num_test_images;
    float* test_image_data;
    float* test_label_data;

    LoadMNISTData(test_image_file, test_label_file, num_test_images, num_channels, height, width, num_classes, &test_image_data, &test_label_data);

    // Evaluate the model on the test dataset
    float accuracy = 0.0f;
    for (int i = 0; i < num_test_images; ++i) {
        std::vector<float> test_input(test_image_data + i * height * width, test_image_data + (i + 1) * height * width);
        std::vector<float> predictions = model.predict(test_input);

        int predicted_class = std::max_element(predictions.begin(), predictions.end()) - predictions.begin();
        int true_class = std::max_element(test_label_data + i * num_classes, test_label_data + (i + 1) * num_classes) - (test_label_data + i * num_classes);

        if (predicted_class == true_class) {
            accuracy += 1.0f;
        }
    }
    accuracy /= num_test_images;

    std::cout << "Test Accuracy: " << accuracy << std::endl;

    // Free the test dataset
    delete[] test_image_data;
    delete[] test_label_data;

    return 0;
}
