import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_quantize(k):
    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, inputs):
            if k == 32:
                output = inputs
            elif k == 1:
                output = torch.sign(inputs)
            else:
                # Quantizes a real number input[0,1] to a k-bit number output[0,1]
                n = float(2 ** k - 1)
                output = torch.round(inputs * n) / n
            return output

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply


# Low bitwidth quantization of weights
class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            # limit the value range of 'weight' to [0,1] and then quantizaing to k-bit 'weight_q'
            weight = torch.tanh(x)
            weight = weight / (2 * torch.max(torch.abs(weight))) + 0.5
            weight_q = 2 * self.uniform_q(weight) - 1
        return weight_q


# Low bitwidth quantization of activation
class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            # Apply an STE on input activations r[0,1]
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
        return activation_q


# Low bitwidth quantization of gradients:

def net_grad_qn(module, gbit):
    def gradients_quantize_fn(dr, gbit):
        m = 2 * torch.max(torch.abs(dr))
        n = float(2 ** gbit - 1)
        nk = torch.empty(dr.size()).uniform_(-0.5, 0.5).cuda()
        # input ri[0,1] to a k-bit number output ro[0,1]
        ri = dr / m + 0.5 #+ nk
        ro = torch.round(ri * n) / n
        dr = m * (ro - 0.5)
        return dr.float()

    def hook_grad_backward(module, grad_input, grad_output):
        return tuple(gradients_quantize_fn(dr, gbit) for dr in grad_input)

    handle_grad_list = []
    handle_grad = module.features[4].register_backward_hook(hook_grad_backward)

    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if name not in ['features.0']:
                handle_grad = m.register_backward_hook(hook_grad_backward)
                handle_grad_list.append(handle_grad)

    return module, handle_grad_list


# conv2d quantization function
def conv2d_Q_fn(w_bit):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=True):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, inputs, order=None):
            weight_q = self.quantize_fn(self.weight)
            return F.conv2d(inputs, weight_q, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    return Conv2d_Q


# linear quantization function
def linear_Q_fn(w_bit):
    class Linear_Q(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, inputs):
            weight_q = self.quantize_fn(self.weight)
            return F.linear(inputs, weight_q, self.bias)

    return Linear_Q
