#include "NeuralNetwork/NeuralNetwork.h"
#include "DataLoader/DataLoader.h"
#include "FeatureExtractor/FeatureExtractor.h"
#include "DataPreprocessor/DataPreprocessor.h"
#include "Metrics/Metrics.h"
#include "FederatedClient/FederatedClient.h"
#include "FederatedServer/FederatedServer.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>

void print_vector(const std::vector<float> &vec, const std::string &label)
{
    std::cout << label << ": [";
    for (size_t i = 0; i < vec.size(); i++)
    {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

void evaluate_client(size_t client_idx, FederatedClient &client,
                     const TrainingSample &test_sample)
{
    auto pred = client.predict(test_sample.features);
    std::cout << "Client " << client_idx << ":\n";
    print_vector(pred, "Prediction");
}

void train_clients_iid(std::vector<std::unique_ptr<FederatedClient>> &clients,
                       std::shared_ptr<DataPreprocessor> preprocessor,
                       float learning_rate,
                       size_t samples_per_client)
{
    for (size_t i = 0; i < samples_per_client; i++)
    {
        // Get balanced batch
        auto balanced_batch = preprocessor->get_balanced_batch(3); // 1 sample per class

        if (!balanced_batch.empty())
        {
            // Each client gets a random sample from this balanced set
            for (auto &client : clients)
            {
                size_t random_idx = rand() % balanced_batch.size();
                const auto &sample = balanced_batch[random_idx];
                client->train_on_sample(sample.features, sample.target, learning_rate);
            }
        }
    }
}

void train_clients_online(std::vector<std::unique_ptr<FederatedClient>> &clients,
                          std::shared_ptr<DataPreprocessor> preprocessor,
                          float learning_rate,
                          size_t samples_per_client)
{
    // Each client gets its own sequence of samples
    for (size_t i = 0; i < samples_per_client; i++)
    {
        // Each client gets a different sample
        for (size_t client_idx = 0; client_idx < clients.size(); client_idx++)
        {
            TrainingSample sample = preprocessor->get_next_training_sample(client_idx);
            clients[client_idx]->train_on_sample(sample.features, sample.target, learning_rate);
        }
    }
}

int main()
{
    try
    {
        // Load dataset
        DataLoader loader("../data");
        auto dataset = loader.load_dataset("motion_metadata.csv");
        std::cout << "Loaded " << dataset.size() << " samples\n\n";

        // Prepare data for training
        auto preprocessor = std::make_shared<DataPreprocessor>(42); // Use consistent seed
        preprocessor->prepare_dataset(dataset);

        // Create neural network topology
        std::vector<size_t> topology = {11, 90, 3};

        // Create federated components
        const size_t NUM_CLIENTS = 1000;
        const size_t TRAINING_SAMPLES_PER_ROUND = 10; // Each client will see this many samples
        const float LEARNING_RATE = 0.7f;
        const int FL_ROUNDS = 500;
        const float CLIENT_FRACTION = 0.01f;

        FederatedServer server;
        std::vector<std::unique_ptr<FederatedClient>> clients;

        // Initialize clients
        for (size_t i = 0; i < NUM_CLIENTS; i++)
        {
            clients.push_back(std::make_unique<FederatedClient>(topology, preprocessor, 42));
        }

        // Get a test sample for evaluation
        auto test_samples = preprocessor->get_test_set();
        if (!test_samples.empty())
        {
            const auto &test_sample = test_samples[0];

            std::cout << "\nTarget values for evaluation:\n";
            print_vector(test_sample.target, "Target   ");

            // Federated Learning Rounds
            for (int round = 0; round < FL_ROUNDS; round++)
            {
                std::cout << "\n=== Federated Learning Round " << (round + 1) << " ===\n";

                // Select subset of clients for this round
                auto selected_clients = server.select_clients(clients.size(), CLIENT_FRACTION);
                std::cout << "Selected " << selected_clients.size() << " clients for this round\n";

                // Local training on selected clients
                std::cout << "\nLocal training with " << TRAINING_SAMPLES_PER_ROUND
                          << " samples per client...\n";

                // Train selected clients
                for (size_t client_idx : selected_clients)
                {
                    for (size_t i = 0; i < TRAINING_SAMPLES_PER_ROUND; i++)
                    {
                        TrainingSample sample = preprocessor->get_next_training_sample(client_idx);
                        clients[client_idx]->train_on_sample(sample.features, sample.target, LEARNING_RATE);
                    }
                }

                // Collect weights only from selected clients
                std::vector<std::vector<float>> client_weights;
                for (size_t client_idx : selected_clients)
                {
                    client_weights.push_back(clients[client_idx]->get_weights());
                }

                // Average weights from selected clients
                auto averaged_weights = server.average_weights(client_weights);

                // Update ALL clients with averaged weights
                for (auto &client : clients)
                {
                    client->set_weights(averaged_weights);
                }

                // Evaluate after weight averaging
                std::cout << "\nPredictions after weight averaging:\n";
                evaluate_client(0, *clients[0], test_sample);
                std::cout << "(All clients now have identical predictions)\n";
            }
            std::cout << "\nTarget values for evaluation:\n";
            print_vector(test_sample.target, "Target   ");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}