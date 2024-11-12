#Adapted from Microsoft/EdgeML
import os
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

import utils

try:
    if utils.findCUDA() is not None:
        import fastgrnn_cuda
except:
    print("Running without FastGRNN CUDA")
    pass
#todo: build and install cuda/setup.py

def onnx_exportable_rnn(input, fargs, cell, output):
    class RNNSymbolic(Function):
        @staticmethod
        def symbolic(g, *fargs):
            # NOTE: args/kwargs contain RNN parameters
            return g.op(cell.name, *fargs,
                        outputs=1, hidden_size_i=cell.state_size,
                        wRank_i=cell.wRank, uRank_i=cell.uRank,
                        gate_nonlinearity_s=cell.gate_nonlinearity,
                        update_nonlinearity_s=cell.update_nonlinearity)

        @staticmethod
        def forward(ctx, *fargs):
            return output

        @staticmethod
        def backward(ctx, *gargs, **gkwargs):
            raise RuntimeError("FIXME: Traced RNNs don't support backward")

    return RNNSymbolic.apply(input, *fargs)

def gen_nonlinearity(A, nonlinearity):
    '''
    Returns required activation for a tensor based on the inputs

    nonlinearity is either a callable or a value in
        ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4']
    '''
    if nonlinearity == "tanh":
        return torch.tanh(A)
    elif nonlinearity == "sigmoid":
        return torch.sigmoid(A)
    elif nonlinearity == "relu":
        return torch.relu(A, 0.0)
    elif nonlinearity == "quantTanh":
        return torch.max(torch.min(A, torch.ones_like(A)), -1.0 * torch.ones_like(A))
    elif nonlinearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    elif nonlinearity == "quantSigm4":
        A = (A + 2.0) / 4.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    else:
        # nonlinearity is a user specified function
        if not callable(nonlinearity):
            raise ValueError("nonlinearity is either a callable or a value " +
                             "['tanh', 'sigmoid', 'relu', 'quantTanh', " +
                             "'quantSigm'")
        return nonlinearity(A)

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size,
                 gate_nonlinearity, update_nonlinearity,
                 num_W_matrices, num_U_matrices, num_biases,
                 wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0):
        super(RNNCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._gate_nonlinearity = gate_nonlinearity
        self._update_nonlinearity = update_nonlinearity
        self._num_W_matrices = num_W_matrices
        self._num_U_matrices = num_U_matrices
        self._num_biases = num_biases
        self._num_weight_matrices = [self._num_W_matrices, self._num_U_matrices,
                                     self._num_biases]
        self._wRank = wRank
        self._uRank = uRank
        self._wSparsity = wSparsity
        self._uSparsity = uSparsity
        self.oldmats = []


    @property
    def state_size(self):
        return self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_nonlinearity(self):
        return self._gate_nonlinearity

    @property
    def update_nonlinearity(self):
        return self._update_nonlinearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_W_matrices(self):
        return self._num_W_matrices

    @property
    def num_U_matrices(self):
        return self._num_U_matrices

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        raise NotImplementedError()

    def forward(self, input, state):
        raise NotImplementedError()

    def getVars(self):
        raise NotImplementedError()

    def get_model_size(self):
        '''
        Function to get aimed model size
        '''
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices

        totalnnz = 2  # For Zeta and Nu
        for i in range(0, endW):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._wSparsity)
            mats[i].to(device)
        for i in range(endW, endU):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._uSparsity)
            mats[i].to(device)
        for i in range(endU, len(mats)):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), False)
            mats[i].to(device)
        return totalnnz * 4

    def copy_previous_UW(self):
        mats = self.getVars()
        num_mats = self._num_W_matrices + self._num_U_matrices
        if len(self.oldmats) != num_mats:
            for i in range(num_mats):
                self.oldmats.append(torch.FloatTensor())
        for i in range(num_mats):
            self.oldmats[i] = torch.FloatTensor(mats[i].detach().clone().to(mats[i].device))

    def sparsify(self):
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices
        for i in range(0, endW):
            mats[i] = utils.hardThreshold(mats[i], self._wSparsity)
        for i in range(endW, endU):
            mats[i] = utils.hardThreshold(mats[i], self._uSparsity)
        self.W.data.copy_(mats[0])
        self.U.data.copy_(mats[1])
        # self.copy_previous_UW()

    def sparsifyWithSupport(self):
        mats = self.getVars()
        endU = self._num_W_matrices + self._num_U_matrices
        for i in range(0, endU):
            mats[i] = utils.supportBasedThreshold(mats[i], self.oldmats[i])

class FastGRNNCell(RNNCell):
    '''
    FastGRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)

    wSparsity = intended sparsity of W matrix(ces)
    uSparsity = intended sparsity of U matrix(ces)
    Warning:
    The Cell will not automatically sparsify.
    The user must invoke .sparsify to hard threshold.

    zetaInit = init for zeta, the scale param
    nuInit = init for nu, the translation param

    FastGRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    z_t = gate_nl(Wx_t + Uh_{t-1} + B_g)
    h_t^ = update_nl(Wx_t + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^

    W and U can further parameterised into low rank version by
    W = matmul(W_1, W_2) and U = matmul(U_1, U_2)
    '''

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, zetaInit=1.0, nuInit=-4.0,
                 name="FastGRNN"):
        super(FastGRNNCell, self).__init__(input_size, hidden_size,
                                          gate_nonlinearity, update_nonlinearity,
                                          1, 1, 2, wRank, uRank, wSparsity,
                                          uSparsity)
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W1 = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U1 = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))
        self.zeta = nn.Parameter(self._zetaInit * torch.ones([1, 1]))
        self.nu = nn.Parameter(self._nuInit * torch.ones([1, 1]))

        # self.copy_previous_UW()

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastGRNN"

    def forward(self, input, state):
        device = self.W.device
        input = input.to(device)
        state = state.to(device)
        if self._wRank is None:
            wComp = torch.matmul(input, self.W)
        else:
            wComp = torch.matmul(
                torch.matmul(input, self.W1), self.W2)

        if self._uRank is None:
            uComp = torch.matmul(state, self.U)
        else:
            uComp = torch.matmul(
                torch.matmul(state, self.U1), self.U2)

        pre_comp = wComp + uComp
        z = gen_nonlinearity(pre_comp + self.bias_gate,
                              self._gate_nonlinearity)
        c = gen_nonlinearity(pre_comp + self.bias_update,
                              self._update_nonlinearity)
        new_h = z * state + (torch.sigmoid(self.zeta) *
                             (1.0 - z) + torch.sigmoid(self.nu)) * c

        return new_h

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_U_matrices == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update])
        Vars.extend([self.zeta, self.nu])
        return Vars

class FastGRNNCUDACell(RNNCell):
    '''
    A CUDA implementation of FastGRNN Cell with Full Rank Support
    hidden_size = # hidden units

    zetaInit = init for zeta, the scale param
    nuInit = init for nu, the translation param

    FastGRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    z_t = non_linearity(Wx_t + Uh_{t-1} + B_g)
    h_t^ = tanh(Wx_t + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^

    '''
    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
    update_nonlinearity="tanh", wRank=None, uRank=None, zetaInit=1.0, nuInit=-4.0, wSparsity=1.0, uSparsity=1.0, name="FastGRNNCUDACell"):
        super(FastGRNNCUDACell, self).__init__(input_size, hidden_size, gate_nonlinearity, update_nonlinearity,
                                                1, 1, 2, wRank, uRank, wSparsity, uSparsity)
        if utils.findCUDA() is None:
            raise Exception('FastGRNNCUDA is supported only on GPU devices.')
        NON_LINEARITY = {"sigmoid": 0, "relu": 1, "tanh": 2}
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        self._name = name
        self.device = torch.device("cuda")

        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W = nn.Parameter(0.1 * torch.randn([hidden_size, input_size], device=self.device))
            self.W1 = torch.empty(0)
            self.W2 = torch.empty(0)
        else:
            self.W = torch.empty(0)
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, input_size], device=self.device))
            self.W2 = nn.Parameter(0.1 * torch.randn([hidden_size, wRank], device=self.device))

        if uRank is None:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, hidden_size], device=self.device))
            self.U1 = torch.empty(0)
            self.U2 = torch.empty(0)
        else:
            self.U = torch.empty(0)
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size], device=self.device))
            self.U2 = nn.Parameter(0.1 * torch.randn([hidden_size, uRank], device=self.device))

        self._gate_non_linearity = NON_LINEARITY[gate_nonlinearity]

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size], device=self.device))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size], device=self.device))
        self.zeta = nn.Parameter(self._zetaInit * torch.ones([1, 1], device=self.device))
        self.nu = nn.Parameter(self._nuInit * torch.ones([1, 1], device=self.device))

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastGRNNCUDACell"

    def forward(self, input, state):
        # Calls the custom autograd function while invokes the CUDA implementation
        if not input.is_cuda:
            input.to(self.device)
        if not state.is_cuda:
            state.to(self.device)
        return FastGRNNFunction.apply(input, self.bias_gate, self.bias_update, self.zeta, self.nu, state,
            self.W, self.U, self.W1, self.W2, self.U1, self.U2, self._gate_non_linearity)

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_U_matrices == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update, self.zeta, self.nu])
        return Vars

class BaseRNN(nn.Module):
    '''
    Generic equivalent of static_rnn in tf
    Used to unroll all the cell written in this file
    We assume batch_first to be False by default
    (following the convention in pytorch) ie.,
    [timeSteps, batchSize, inputDims] else
    [batchSize, timeSteps, inputDims]
    '''

    def __init__(self, cell: RNNCell, batch_first=False, cell_reverse: RNNCell=None, bidirectional=False):
        super(BaseRNN, self).__init__()
        self.RNNCell = cell
        self._batch_first = batch_first
        self._bidirectional = bidirectional
        if cell_reverse is not None:
            self.RNNCell_reverse = cell_reverse
        elif self._bidirectional:
            self.RNNCell_reverse = cell

    def getVars(self):
        return self.RNNCell.getVars()

    def forward(self, input, hiddenState=None,
                cellState=None):
        self.device = input.device
        self.num_directions = 2 if self._bidirectional else 1
        # hidden
        # for i in range(num_directions):
        hiddenStates = torch.zeros(
                [input.shape[0], input.shape[1],
                 self.RNNCell.output_size]).to(self.device)

        if self._bidirectional:
                hiddenStates_reverse = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell_reverse.output_size]).to(self.device)

        if hiddenState is None:
                hiddenState = torch.zeros(
                    [self.num_directions, input.shape[0] if self._batch_first else input.shape[1],
                    self.RNNCell.output_size]).to(self.device)

        if self._batch_first is True:
            if self.RNNCell.cellType == "LSTMLR":
                cellStates = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell.output_size]).to(self.device)
                if self._bidirectional:
                    cellStates_reverse = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell_reverse.output_size]).to(self.device)
                if cellState is None:
                    cellState = torch.zeros(
                        [self.num_directions, input.shape[0], self.RNNCell.output_size]).to(self.device)
                for i in range(0, input.shape[1]):
                    hiddenState[0], cellState[0] = self.RNNCell(
                        input[:, i, :], (hiddenState[0].clone(), cellState[0].clone()))
                    hiddenStates[:, i, :] = hiddenState[0]
                    cellStates[:, i, :] = cellState[0]
                    if self._bidirectional:
                        hiddenState[1], cellState[1] = self.RNNCell_reverse(
                            input[:, input.shape[1]-i-1, :], (hiddenState[1].clone(), cellState[1].clone()))
                        hiddenStates_reverse[:, i, :] = hiddenState[1]
                        cellStates_reverse[:, i, :] = cellState[1]
                if not self._bidirectional:
                    return hiddenStates, cellStates
                else:
                    return torch.cat([hiddenStates,hiddenStates_reverse],-1), torch.cat([cellStates,cellStates_reverse],-1)
            else:
                for i in range(0, input.shape[1]):
                    hiddenState[0] = self.RNNCell(input[:, i, :], hiddenState[0].clone())
                    hiddenStates[:, i, :] = hiddenState[0]
                    if self._bidirectional:
                        hiddenState[1] = self.RNNCell_reverse(
                            input[:, input.shape[1]-i-1, :], hiddenState[1].clone())
                        hiddenStates_reverse[:, i, :] = hiddenState[1]
                if not self._bidirectional:
                    return hiddenStates
                else:
                    return torch.cat([hiddenStates,hiddenStates_reverse],-1)
        else:
            if self.RNNCell.cellType == "LSTMLR":
                cellStates = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell.output_size]).to(self.device)
                if self._bidirectional:
                    cellStates_reverse = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell_reverse.output_size]).to(self.device)
                if cellState is None:
                    cellState = torch.zeros(
                        [self.num_directions, input.shape[1], self.RNNCell.output_size]).to(self.device)
                for i in range(0, input.shape[0]):
                    hiddenState[0], cellState[0] = self.RNNCell(
                        input[i, :, :], (hiddenState[0].clone(), cellState[0].clone()))
                    hiddenStates[i, :, :] = hiddenState[0]
                    cellStates[i, :, :] = cellState[0]
                    if self._bidirectional:
                        hiddenState[1], cellState[1] = self.RNNCell_reverse(
                            input[input.shape[0]-i-1, :, :], (hiddenState[1].clone(), cellState[1].clone()))
                        hiddenStates_reverse[i, :, :] = hiddenState[1]
                        cellStates_reverse[i, :, :] = cellState[1]
                if not self._bidirectional:
                    return hiddenStates, cellStates
                else:
                    return torch.cat([hiddenStates,hiddenStates_reverse],-1), torch.cat([cellStates,cellStates_reverse],-1)
            else:
                for i in range(0, input.shape[0]):
                    hiddenState[0] = self.RNNCell(input[i, :, :], hiddenState[0].clone())
                    hiddenStates[i, :, :] = hiddenState[0]
                    if self._bidirectional:
                        hiddenState[1] = self.RNNCell_reverse(
                            input[input.shape[0]-i-1, :, :], hiddenState[1].clone())
                        hiddenStates_reverse[i, :, :] = hiddenState[1]
                if not self._bidirectional:
                    return hiddenStates
                else:
                    return torch.cat([hiddenStates,hiddenStates_reverse],-1)

class FastGRNN(nn.Module):
    """Equivalent to nn.FastGRNN using FastGRNNCell"""

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, zetaInit=1.0, nuInit=-4.0,
                 batch_first=False, bidirectional=False, is_shared_bidirectional=True):
        super(FastGRNN, self).__init__()
        self._bidirectional = bidirectional
        self._batch_first = batch_first
        self._is_shared_bidirectional = is_shared_bidirectional
        self.cell = FastGRNNCell(input_size, hidden_size,
                                 gate_nonlinearity=gate_nonlinearity,
                                 update_nonlinearity=update_nonlinearity,
                                 wRank=wRank, uRank=uRank,
                                 wSparsity=wSparsity, uSparsity=uSparsity,
                                 zetaInit=zetaInit, nuInit=nuInit)
        self.unrollRNN = BaseRNN(self.cell, batch_first=self._batch_first, bidirectional=self._bidirectional)

        if self._bidirectional is True and self._is_shared_bidirectional is False:
            self.cell_reverse = FastGRNNCell(input_size, hidden_size,
                                 gate_nonlinearity=gate_nonlinearity,
                                 update_nonlinearity=update_nonlinearity,
                                 wRank=wRank, uRank=uRank,
                                 wSparsity=wSparsity, uSparsity=uSparsity,
                                 zetaInit=zetaInit, nuInit=nuInit)
            self.unrollRNN = BaseRNN(self.cell, self.cell_reverse, batch_first=self._batch_first, bidirectional=self._bidirectional)

    def getVars(self):
        return self.unrollRNN.getVars()

    def forward(self, input, hiddenState=None, cellState=None):
        return self.unrollRNN(input, hiddenState, cellState)

class FastGRNNCUDA(nn.Module):
    """
        Unrolled implementation of the FastGRNNCUDACell
        Note: update_nonlinearity is fixed to tanh, only gate_nonlinearity
        is configurable.
    """
    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 wSparsity=1.0, uSparsity=1.0, zetaInit=1.0, nuInit=-4.0,
                 batch_first=False, name="FastGRNNCUDA"):
        super(FastGRNNCUDA, self).__init__()
        if utils.findCUDA() is None:
            raise Exception('FastGRNNCUDA is supported only on GPU devices.')
        NON_LINEARITY = {"sigmoid": 0, "relu": 1, "tanh": 2}
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        self._name = name
        self._num_W_matrices = 1
        self._num_U_matrices = 1
        self._num_biases = 2
        self._num_weight_matrices = [self._num_W_matrices, self._num_U_matrices, self._num_biases]
        self._wRank = wRank
        self._uRank = uRank
        self._wSparsity = wSparsity
        self._uSparsity = uSparsity
        self.oldmats = []
        self.device = torch.device("cuda")
        self.batch_first = batch_first
        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name

        if wRank is None:
            self.W = nn.Parameter(0.1 * torch.randn([hidden_size, input_size], device=self.device))
            self.W1 = torch.empty(0)
            self.W2 = torch.empty(0)
        else:
            self.W = torch.empty(0)
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, input_size], device=self.device))
            self.W2 = nn.Parameter(0.1 * torch.randn([hidden_size, wRank], device=self.device))

        if uRank is None:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, hidden_size], device=self.device))
            self.U1 = torch.empty(0)
            self.U2 = torch.empty(0)
        else:
            self.U = torch.empty(0)
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size], device=self.device))
            self.U2 = nn.Parameter(0.1 * torch.randn([hidden_size, uRank], device=self.device))

        self._gate_non_linearity = NON_LINEARITY[gate_nonlinearity]

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size], device=self.device))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size], device=self.device))
        self.zeta = nn.Parameter(self._zetaInit * torch.ones([1, 1], device=self.device))
        self.nu = nn.Parameter(self._nuInit * torch.ones([1, 1], device=self.device))

    def forward(self, input, hiddenState=None, cell_state=None):
        '''
            input: [timesteps, batch, features]; hiddenState: [batch, state_size]
            hiddenState is set to zeros if not provided.
        '''
        if self.batch_first is True:
            input = input.transpose(0, 1).contiguous()
        if not input.is_cuda:
            input = input.to(self.device)
        if hiddenState is None:
            hiddenState = torch.zeros(
                [input.shape[1], self._hidden_size]).to(self.device)
        if not hiddenState.is_cuda:
            hiddenState = hiddenState.to(self.device)
        result = FastGRNNUnrollFunction.apply(input, self.bias_gate, self.bias_update, self.zeta, self.nu, hiddenState,
            self.W, self.U, self.W1, self.W2, self.U1, self.U2, self._gate_non_linearity)
        if self.batch_first is True:
            return result.transpose(0, 1)
        else:
            return result

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_U_matrices == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update, self.zeta, self.nu])
        return Vars

    def get_model_size(self):
        '''
        Function to get aimed model size
        '''
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices

        totalnnz = 2  # For Zeta and Nu
        for i in range(0, endW):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._wSparsity)
            mats[i].to(device)
        for i in range(endW, endU):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._uSparsity)
            mats[i].to(device)
        for i in range(endU, len(mats)):
            device = mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), False)
            mats[i].to(device)
        return totalnnz * 4

    def copy_previous_UW(self):
        mats = self.getVars()
        num_mats = self._num_W_matrices + self._num_U_matrices
        if len(self.oldmats) != num_mats:
            for i in range(num_mats):
                self.oldmats.append(torch.FloatTensor())
        for i in range(num_mats):
            self.oldmats[i] = torch.FloatTensor(mats[i].detach().clone().to(mats[i].device))

    def sparsify(self):
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices
        for i in range(0, endW):
            mats[i] = utils.hardThreshold(mats[i], self._wSparsity)
        for i in range(endW, endU):
            mats[i] = utils.hardThreshold(mats[i], self._uSparsity)
        self.copy_previous_UW()

    def sparsifyWithSupport(self):
        mats = self.getVars()
        endU = self._num_W_matrices + self._num_U_matrices
        for i in range(0, endU):
            mats[i] = utils.supportBasedThreshold(mats[i], self.oldmats[i])

class FastGRNNFunction(Function):
    @staticmethod
    def forward(ctx, input, bias_gate, bias_update, zeta, nu, old_h, w, u, w1, w2, u1, u2, gate_non_linearity):
        outputs = fastgrnn_cuda.forward(input.contiguous(), w, u, bias_gate, bias_update, zeta, nu, old_h, gate_non_linearity, w1, w2, u1, u2)
        new_h = outputs[0]
        variables = [input, old_h, zeta, nu, w, u] + outputs[1:] + [w1, w2, u1, u2]
        ctx.save_for_backward(*variables)
        ctx.non_linearity = gate_non_linearity
        return new_h

    @staticmethod
    def backward(ctx, grad_h):
        outputs = fastgrnn_cuda.backward(
            grad_h.contiguous(), *ctx.saved_variables, ctx.non_linearity)
        return tuple(outputs + [None])

class FastGRNNUnrollFunction(Function):
    @staticmethod
    def forward(ctx, input, bias_gate, bias_update, zeta, nu, old_h, w, u, w1, w2, u1, u2, gate_non_linearity):
        outputs = fastgrnn_cuda.forward_unroll(input.contiguous(), w, u, bias_gate, bias_update, zeta, nu, old_h, gate_non_linearity, w1, w2, u1, u2)
        hidden_states = outputs[0]
        variables = [input, hidden_states, zeta, nu, w, u] + outputs[1:] + [old_h, w1, w2, u1, u2]
        ctx.save_for_backward(*variables)
        ctx.gate_non_linearity = gate_non_linearity
        return hidden_states

    @staticmethod
    def backward(ctx, grad_h):
        outputs = fastgrnn_cuda.backward_unroll(
            grad_h.contiguous(), *ctx.saved_variables, ctx.gate_non_linearity)
        return tuple(outputs + [None])