#include "hsd_model.h"


using namespace std;


HSDModel::HSDModel(size_t histLen, size_t numFields, size_t embedDim)
{
    m_histEncoder = register_module("m_histEncoder",
        torch::nn::GRU(torch::nn::GRUOptions(numFields, embedDim)
                                         .num_layers(3)
                                         .batch_first(true)
                                         .bidirectional(false)
                                         .dropout(0.1)));

    m_targetEmbedder = register_module("m_targetEmbedder",
        torch::nn::Linear(1, embedDim)
    );

    m_jointEncoder = register_module("m_jointEncoder",
        torch::nn::Sequential(
            torch::nn::ReLU(),
            torch::nn::Dropout(0.1),
            torch::nn::Linear(embedDim * 2, embedDim * 2),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.1),
            torch::nn::Linear(embedDim * 2, embedDim),
            torch::nn::ReLU(),
            torch::nn::Linear(embedDim, 2)
        )
    );
}

HSDOutput HSDModel::forward(torch::Tensor histData, torch::Tensor dateTarget)
{
    auto encAll = m_histEncoder(histData);

    torch::Tensor histEnc = get<1>(encAll).select(0, -1);

    torch::Tensor dateEmbed = m_targetEmbedder(dateTarget);

    torch::Tensor jointInput = torch::cat({histEnc, dateEmbed}, 1);

    torch::Tensor output = m_jointEncoder->forward(jointInput);

    auto ops = output.split(1, 1);

    return { ops[0], ops[1] };
}