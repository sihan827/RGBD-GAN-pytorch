import torch


class GradientScaler(torch.autograd.Function):
    factor = 1.0

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        factor = GradientScaler.factor
        return factor.view(-1, 1, 1, 1) * grad_output


def get_gradient_ratios(lossA, lossB, x_f, eps=1e-6):
    grad_lossA_xf = torch.autograd.grad(torch.sum(lossA), x_f, retain_graph=True)[0]
    grad_lossB_xf = torch.autograd.grad(torch.sum(lossB), x_f, retain_graph=True)[0]
    gamma = grad_lossA_xf / grad_lossB_xf

    return gamma
