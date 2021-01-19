#include "hsd_loss.h"

namespace nn = torch::nn;
namespace F = nn::functional;


LossInfo HSDLoss::forward(const HSDOutput &x, torch::Tensor y)
{
    auto mse = F::mse_loss(x.ClosingPrice, y, torch::kNone);

    auto uncSq = x.Certainty * x.Certainty;

    // auto loss = (mse / (2 * uncSq)) + 0.5 * uncSq.log();
    auto loss = mse;

    LossInfo ret;
    ret.Loss = loss.mean();

    ret.Components = {
        { "MSE", mse.detach().mean() },
        { "Uncertainty", uncSq.detach().mean() }
    };

    return ret;
}