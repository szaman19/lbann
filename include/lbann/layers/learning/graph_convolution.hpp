////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_LAYERS_LEARNING_GRAPH_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_GRAPH_CONVOLUTION_HPP_INCLUDED

#include "lbann/layers/learning/learning.hpp"
#include "lbann/models/model.hpp"

namespace lbann{

/** @brief Graph Convolution 
 *
 * The input is a node feature tensor of shape (N x M) and an Adjacency Matrix of shape 
 * N x N. This performs the mat-mat multiplicatin: 
 *   @f[X` = AXW + B@f]
 * 
 * Two weights are required if bias is applied. If weights aren't provided, The linearity 
 * weights are initialized with He normal initialization and the bias weights are initialized
 * to zero. 
 **/

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class graph_convolution_layer: public learning_layer<TensorDataType>{

/** @name Public Types */ 

///@{
  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;
  
  /** @brief The concrete weights type used by this object. */ 
  using WeightsType = data_type_weights<TensorDataType>;

  /** @brief The concrete optimizer type used by this object */ 
  using optimizerType = data_type_optimizer<TensorDataType>;
///@}
public:
  graph_convolution_layer(lbann_comm *comm,
                          int output_channel,
                          WeightsType* weight = nullptr,
                          bool has_bias = true); //Constructor 
  graph_convolution_layer(const graph_convolution_layer& other); //Copy-Constructor 

  graph_convolution_layer& operator= (const graph_convolution_layer& other);

  ~graph_convolution_layer() override;

  graph_convolution_layer* copy() override{
    return new graph_convolution_layer(*this);
  }

  std::string get_type() const override {return "graph convolution"}
  data_layout get_data_layout() const override {return T_layout; }
  El::Device get_device_allocation() const override {return Dev; }

  description get_description() const override; 

protected:
  void setup_matrices(const El:Grid& grid) override; 
  void setup_data(size_t max_mini_batch_size) override; 
  void fp_compute(); //Forward-Propagation
  void bp_compute(); //Backward-Propagation

private:
  /** Scaling factor for bias term. 
   * If the scaling factor is xzero, bias is not applied.
   *
   */
  TensorDataType m_bias_scaling_factor; 
  /** Bias weights gradient.
   *
   */
  AbsDistMatrixType * mbias_gradient; 
  
  /** Deallocate distributed matrices. */
  void deallocate_matrices(){
    if(m_bias_gradient != nullptr){
      delete m_bias_gradient;
    }
  }

  template <typename U>
  friend void fp_compute_impl(graph_convolution_layer<U, T_layout, Dev>& l);
  template <typename U>
  friend void bp_compute_impl(graph_convolution_layer<U, T_layout, Dev>& l);
};

//Builder function 

LBANN_DEFINE_LAYER_BUILDER(graph_convolution);

#ifndef LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device) \ 
  extern template class graph_convolution_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class graph_convolution_layer<T, data_layout::MODEL_PARALLEL, Device>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE

} // namespace lbann 

#endif // LBANN_LAYERS_LEARNING_GRAPH_CONVOLUTION_HPP_INCLUDED
