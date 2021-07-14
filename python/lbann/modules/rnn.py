"""Neural network modules for recurrent models."""

import math
import lbann
from .base import Module, FullyConnectedModule
from lbann.util import make_iterable, str_list

class LSTMCell(Module):
    """Long short-term memory cell."""

    global_count = 0  # Static counter, used for default names

    def __init__(self, size, bias = True,
                 weights=[], name=None, data_layout='data_parallel'):
        """Initialize LSTM cell.

        Args:
            size (int): Size of output tensor.
            bias (bool): Whether to apply biases after linearity.
            weights (`Weights` or iterator of `Weights`): Weights in
                fully-connected layer. There are at most two - a
                matrix ((4*size) x (input_size+size) dimensions) and a
                bias (4*size entries). If weights are not provided,
                the matrix and bias will be initialized in a similar
                manner as PyTorch (uniform random values from
                [-1/sqrt(size), 1/sqrt(size)]).
            name (str): Default name is in the form 'lstmcell<index>'.
            data_layout (str): Data layout.

        """
        super().__init__()
        LSTMCell.global_count += 1
        self.step = 0
        self.size = size
        self.name = (name
                     if name
                     else 'lstmcell{0}'.format(LSTMCell.global_count))
        self.data_layout = data_layout

        # Weights
        self.weights = list(make_iterable(weights))
        if len(self.weights) > 2:
            raise ValueError('`LSTMCell` has at most two weights, '
                             'but got {0}'.format(len(self.weights)))
        scale = 1 / math.sqrt(self.size)
        if len(self.weights) == 0:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-scale,
                                                                   max=scale),
                              name=self.name+'_matrix')
            )
        if len(self.weights) == 1:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-scale,
                                                                   max=scale),
                              name=self.name+'_bias')
            )

        # Linearity
        self.fc = FullyConnectedModule(
            4*size, bias=bias,
            weights=self.weights,
            name=self.name + '_fc',
            data_layout=self.data_layout
        )

    def forward(self, x, prev_state):
        """Apply LSTM step.

        Args:
            x (Layer): Input.
            prev_state (tuple with two `Layer`s): State from previous
                LSTM step. Comprised of LSTM output and cell state.

        Returns:
            (Layer, (Layer, Layer)): The output and state (the output
                and cell state). The state can be passed directly into
                the next LSTM step.

        """
        self.step += 1
        name = '{0}_step{1}'.format(self.name, self.step)

        # Get output and cell state from previous step
        prev_output, prev_cell = prev_state

        # Apply linearity
        input_concat = lbann.Concatenation(x, prev_output,
                                           name=name + '_input',
                                           data_layout=self.data_layout)
        fc = self.fc(input_concat)

        # Get gates and cell update
        slice = lbann.Slice(fc,
                            slice_points=str_list([0, self.size, 4*self.size]),
                            name=name + '_fc_slice',
                            data_layout=self.data_layout)
        cell_update = lbann.Tanh(slice,
                                 name=name + '_cell_update',
                                 data_layout=self.data_layout)
        sigmoid = lbann.Sigmoid(slice,
                                name=name + '_sigmoid',
                                data_layout=self.data_layout)
        slice = lbann.Slice(sigmoid,
                            slice_points=str_list([0, self.size, 2*self.size, 3*self.size]),
                            name=name + '_sigmoid_slice',
                            data_layout=self.data_layout)
        f = lbann.Identity(slice, name=name + '_forget_gate',
                           data_layout=self.data_layout)
        i = lbann.Identity(slice, name=name + '_input_gate',
                           data_layout=self.data_layout)
        o = lbann.Identity(slice, name=name + '_output_gate',
                           data_layout=self.data_layout)

        # Cell state
        cell_forget = lbann.Multiply(f, prev_cell,
                                     name=name + '_cell_forget',
                                     data_layout=self.data_layout)
        cell_input = lbann.Multiply(i, cell_update,
                                    name=name + '_cell_input',
                                    data_layout=self.data_layout)
        cell = lbann.Add(cell_forget, cell_input, name=name + '_cell',
                         data_layout=self.data_layout)

        # Output
        cell_act = lbann.Tanh(cell, name=name + '_cell_activation',
                              data_layout=self.data_layout)
        output = lbann.Multiply(o, cell_act, name=name,
                                data_layout=self.data_layout)

        # Return output and state
        return output, (output, cell)

class GRU(Module):
    """Gated-recurrent unit.
       Implementation mostly taken from:
       https://pytorch.org/docs/stable/nn.html#gru"""

    global_count = 0  # Static counter, used for default names

    def __init__(self, size, bias = True,
                 weights=[], name=None, data_layout='data_parallel'):
        """Initialize GRU cell.

        Args:
            size (int): Size of output tensor.
            bias (bool): Whether to apply biases after linearity.
            weights (`Weights` or iterator of `Weights`): Weights in
                fully-connected layer. There are at most four - two
                matrices ((3*size) x (input_size) and (3*size) x (size) dimensions) each and two
                biases (3*size entries) each. If weights are not provided,
                the matrix and bias will be initialized in a similar
                manner as PyTorch (uniform random values from
                [-1/sqrt(size), 1/sqrt(size)]).
            name (str): Default name is in the form 'gru<index>'.
            data_layout (str): Data layout.

        """
        super().__init__()
        GRU.global_count += 1
        self.step = 0
        self.size = size
        self.name = (name
                     if name
                     else 'gru{0}'.format(GRU.global_count))
        self.data_layout = data_layout

        # Weights
        self.weights = list(make_iterable(weights))
        if len(self.weights) > 4:
            raise ValueError('`GRU` has at most 4 weights, '
                             'but got {0}'.format(len(self.weights)))
        ##@todo: use loop
        scale = 1 / math.sqrt(self.size)
        if len(self.weights) == 0:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-scale,
                                                                   max=scale),
                              name=self.name+'_ih_matrix')
            )
        if len(self.weights) == 1:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-scale,
                                                                   max=scale),
                              name=self.name+'_ih_bias')
            )
        if len(self.weights) == 2:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-scale,
                                                                   max=scale),
                              name=self.name+'_hh_matrix')
            )
        if len(self.weights) == 3:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-scale,
                                                                   max=scale),
                              name=self.name+'_hh_bias')
            )

        # Linearity
        ####Learnable input-hidden weights
        self.ih_fc = FullyConnectedModule(
            3*size, bias=bias,
            weights=self.weights[:2],
            name=self.name + '_ih_fc',
            data_layout=self.data_layout
        )
        ###Learnable hidden-hidden weights
        self.hh_fc = FullyConnectedModule(
            3*size, bias=bias,
            weights=self.weights[2:],
            name=self.name + '_hh_fc',
            data_layout=self.data_layout
        )

        self.ones = lbann.Constant(
            value=1.0,
            num_neurons=str(size),
            data_layout=self.data_layout,
            name=self.name+'_ones',
        )

    def forward(self, x, prev_state):
        """Apply GRU step.

        Args:
            x (Layer): Input.
            prev_state: State from previous GRU step.

        Returns:
            (Layer, Layer): The output (out)  and state (hn).
                          The state can be passed directly into
                           the next GRU step.

        """
        self.step += 1
        name = '{0}_step{1}'.format(self.name, self.step)


        fc1 = self.ih_fc(x)   #input_fc
        fc2 = self.hh_fc(prev_state)  #hidden_fc


        # Get gates and cell update
        fc1_slice = lbann.Slice(fc1,
                            slice_points=str_list([0, self.size, 2*self.size, 3*self.size]),
                            name=name + '_fc1_slice',
                            data_layout=self.data_layout)
        Wir_x = lbann.Identity(fc1_slice, name=name + '_Wrx',
                           data_layout=self.data_layout)
        Wiz_x = lbann.Identity(fc1_slice, name=name + '_Wzx',
                           data_layout=self.data_layout)
        Win_x = lbann.Identity(fc1_slice, name=name + '_Wnx',
                           data_layout=self.data_layout)

        fc2_slice = lbann.Slice(fc2,
                            slice_points=str_list([0, self.size, 2*self.size, 3*self.size]),
                            name=name + '_fc2_slice',
                            data_layout=self.data_layout)
        Whr_prev = lbann.Identity(fc2_slice, name=name + '_Wrh',
                           data_layout=self.data_layout)
        Whz_prev = lbann.Identity(fc2_slice, name=name + '_Wzh',
                           data_layout=self.data_layout)
        Whn_prev = lbann.Identity(fc2_slice, name=name + '_Wnh',
                           data_layout=self.data_layout)

        rt = \
            lbann.Sigmoid(
                lbann.Add(Wir_x, Whr_prev, data_layout=self.data_layout),
                name=name + '_reset_gate',
                data_layout=self.data_layout
            )

        zt = \
            lbann.Sigmoid(
                lbann.Add(Wiz_x, Whz_prev, data_layout=self.data_layout),
                name=name + '_update_gate',
                data_layout=self.data_layout,
            )

        nt = \
            lbann.Tanh(
                lbann.Add(
                    Win_x,
                    lbann.Multiply(rt, Whn_prev, data_layout=self.data_layout),
                    data_layout=self.data_layout,
                ),
                name=name + '_new_gate', data_layout=self.data_layout,
            )

        ht = \
            lbann.Add(
                lbann.Multiply(
                    lbann.WeightedSum(
                        self.ones,
                        zt,
                        scaling_factors='1 -1', data_layout=self.data_layout
                    ),
                    nt,
                    data_layout=self.data_layout
                ),
                lbann.Multiply(zt, prev_state, data_layout=self.data_layout),
                name=name+ '_output', data_layout=self.data_layout,
            )

        # Return output
        return ht, ht


class ChannelwiseGRU(Module):
    global_count = 0

    def __init__(self, num_channels, size, bias=True, weights=[], name=None):

        super().__init()
        ChannelwiseGRU.global_count += 1 
        self.step = 0
        self.size = size
        self.num_channels = num_channels 
        self.name = (name if name else f'gru{ChannelwiseGRU.global_count}')

        scale = 1 / math.sqrt(self.size)

        self.weights = list(make_iterable(weights))

        weight_name = ['_ih_matrix', '_ih_bias', '_hh_matrix', '_hh_bias']
        for i in range(4):
            if (len(self.weights) == i):
                self.weights.append(lbann.Weights(initializer=lbann.UniformInitializer(min=-scale,
                                                                                       max=scale),
                                                  name=self.name+weight_name[i])
                )

        self.ih_fc = ChannelwiseFullyConnectedModule(3*size,
                                                     bias=bias,
                                                     weights=self.weights[:2],
                                                     self.name=self.name + '_ih_fc')
        self.hh_fc = ChannelwiseFullyConnectedModule(3*size,
                                                     bias=bias,
                                                     weights=self.weights[2:],
                                                     self.name=self.name + '_hh_fc')
        self.ones = lbann.Constant(
            values=1.0,
            num_neurons = str(size * num_channels),
            name=self.name+'_ones')

    def forward(self, x, prev_state):
        """ Apply GRU step channelwise 

        Args: 
            x (Layer): Input (shape: (num_channels, *))
            prev_state (Layer): Sate from previous GRU step  (shape: (num_channels, size))
        Returns:
            (Layer, Layer): The output (out) and state (hn). The state can be passed directly into the next GRU step
        """

        self.step += 1

        name = f"{self.name}_step{self.step}"

        mat_size = self.num_channels * self.size

        prev_state = lbann.Reshape(prev_state, dims=f"{mat_size}", name=name+"_prev_state_reshape")

        fc1 = lbann.Reshape(self.ih_fc(x), dims=f"{mat_size * 3}")
        fc2 = lbann.Reshape(self.hh_fc(prev_state),dims==f"{mat_size * 3}")

        fc1_slice = lbann.Slice(fc1, 
                                slice_points=str_list([0, mat_size, 2*mat_size, 3*mat_size]))
        
        Wir_x =lbann.Identity(fc1_slice, name=name+'_Wx')
        Wiz_z =lbann.Identity(fc1_slice, name=name+'_Wx')
        Win_x =lbann.Identity(fc1_slice, name=name+'_Wx')
        fc2_slice = lbann.Slice(fc2, 
                                slice_points=str_list([0, mat_size, 2*mat_size, 3*mat_size]))

        Whr_x =lbann.Identity(fc2_slice, name=name+'_Wh')   
        Whz_z =lbann.Identity(fc2_slice, name=name+'_Wh') 
        Whn_x =lbann.Identity(fc2_slice, name=name+'_Wh')

        rt = \
            lbann.Sigmoid(
                lbann.Add(Wir_x, Whr_prev, data_layout=self.data_layout),
                name=name + '_reset_gate',
                data_layout=self.data_layout
            )

        zt = \
            lbann.Sigmoid(
                lbann.Add(Wiz_x, Whz_prev, data_layout=self.data_layout),
                name=name + '_update_gate',
                data_layout=self.data_layout,
            )

        nt = \
            lbann.Tanh(
                lbann.Add(
                    Win_x,
                    lbann.Multiply(rt, Whn_prev, data_layout=self.data_layout),
                    data_layout=self.data_layout,
                ),
                name=name + '_new_gate', data_layout=self.data_layout,
            )

        ht = \
            lbann.Add(
                lbann.Multiply(
                    lbann.WeightedSum(
                        self.ones,
                        zt,
                        scaling_factors='1 -1', data_layout=self.data_layout
                    ),
                    nt,
                    data_layout=self.data_layout
                ),
                lbann.Multiply(zt, prev_state, data_layout=self.data_layout),
                name=name+ '_output', data_layout=self.data_layout,
            )

        ht = lbann.Reshape(ht, dims=str_list([self.num_channels, self.size]))

        return ht, ht
