import os
import sys

import torch
from torch import autograd
from torch.utils import cpp_extension

module = sys.modules[__name__]

# path = os.path.join(os.path.dirname(__file__), "extension")

path = os.path.dirname(__file__)
rspmm_sources = [
    # os.path.join(path, "spmm.cpp"),
    os.path.join(path, "rspmm.cpp"),
    # os.path.join(path, "spmm.cu"), 
    os.path.join(path, "rspmm.cu")
    ]

sparse_source = [os.path.join(path, "torch_ext.cpp")]

def load_extension(name, sources):
    extra_cflags = ["-Ofast", "-g"]
    extra_cflags.append("-DAT_PARALLEL_NATIVE")
    extra_cuda_cflags = ["-O3"]
    extra_cflags.append("-DCUDA_OP")
    extra_ldflags = ['-ltorch']
    worker_name = "%s" % (name)
    build_directory = cpp_extension._get_build_directory(worker_name, True)
    return cpp_extension.load(name=name, sources=sources, extra_cflags=extra_cflags, extra_cuda_cflags=extra_cuda_cflags, build_directory=build_directory, extra_ldflags=extra_ldflags, verbose=True)

rspmm = load_extension('rspmm', rspmm_sources)
torch_ext = load_extension("torch_ext", sparse_source)

class RSPMMAddMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_mul_forward_cuda
        else:
            forward = rspmm.rspmm_add_mul_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_mul_backward_cuda
        else:
            backward = rspmm.rspmm_add_mul_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMMinMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_mul_forward_cuda
        else:
            forward = rspmm.rspmm_min_mul_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_mul_backward_cuda
        else:
            backward = rspmm.rspmm_min_mul_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMMaxMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_mul_forward_cuda
        else:
            forward = rspmm.rspmm_max_mul_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_mul_backward_cuda
        else:
            backward = rspmm.rspmm_max_mul_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMAddAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_add_forward_cuda
        else:
            forward = rspmm.rspmm_add_add_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_add_backward_cuda
        else:
            backward = rspmm.rspmm_add_add_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMMinAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_add_forward_cuda
        else:
            forward = rspmm.rspmm_min_add_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_add_backward_cuda
        else:
            backward = rspmm.rspmm_min_add_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMMaxAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_add_forward_cuda
        else:
            forward = rspmm.rspmm_max_add_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_add_backward_cuda
        else:
            backward = rspmm.rspmm_max_add_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad

def sparse_coo_tensor(indices, values, size):
    """
    Construct a sparse COO tensor without index check. Much faster than `torch.sparse_coo_tensor`_.
    .. _torch.sparse_coo_tensor:
        https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
    Parameters:
        indices (Tensor): 2D indices of shape (2, n)
        values (Tensor): values of shape (n,)
        size (list): size of the tensor
    """
    return torch_ext.sparse_coo_tensor_unsafe(indices, values, size)

def generalized_rspmm(sparse, relation, input, sum="add", mul="mul"):
    r"""
    Generalized relational sparse-dense matrix multiplication.
    This function computes the matrix multiplication of a sparse matrix, a dense relation matrix and
    a dense input matrix. The output dense matrix satisfies
    .. math::
        output_{i,l} = \bigoplus_{j,k: sparse_{i,j,k} \neq 0} sparse_{i, j, k} \times (relation_{k,l} \otimes input_{j,l})
    where :math:`\oplus` and :math:`\otimes` are the summation and the multiplication operators respectively.
    .. warning::
        Gradient w.r.t. the sparse matrix is only computed for non-zero entries of the sparse matrix.
        This behaves differently from dense-dense matrix multiplication with zero entries.
    Parameters:
        sparse (SparseTensor): 3D sparse tensor
        relation (Tensor): 2D dense tensor
        input (Tensor): 2D dense tensor
        sum (str, optional): generalized summation operator. Available operators are ``add``, ``min`` and ``max``.
        mul (str, optional): generalized multiplication operator. Available operators are ``add`` and ``mul``.
    """
    name = "RSPMM%s%sFunction" % (sum.capitalize(), mul.capitalize())
    if not hasattr(module, name):
        raise ValueError("No generalized rspmm implementation found for summation `%s` and multiplication `%s`"
                         % (sum, mul))
    Function = getattr(module, name)
    return Function.apply(sparse.coalesce(), relation, input)
