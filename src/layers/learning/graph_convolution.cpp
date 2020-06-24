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

#define LBANN_GRAPH_CONVOLUTION_LAYER_INSTANTIATE
#include "lbann/layers/learning/graph_convolution.hpp"

#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

#include <layers.pb.h>

#include <string>
#include <sstream>

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El: Device Dev>
graph_convolution_layer <TensorDataType, T_layout, Dev> 
:: graph_convolution_layer(
  lbann_comm *comm,
  int output_channel, 
  WeightsType* weight, 
  bool has_bias)
  : learning_layer<TensorDataType>(comm),
  m_bias_gradient(nullptr){
  
  // Initualize output tensor dimensions 
  //this->set_output_dims({}); 
  if has_bias {
    m_bias_scaling_factor = El::TypeTraits<TensorDataType>::One();
  }else{
    m_bias_scalling_factor = El::TypeTraits<TensorDataType>::Zero();
  }
  //The layer should have 2 parents. Node feature and adjacency matrix 
  m_expected_num_parent_layers = 2; 
} // Constructor


// Setup layer data and allocate memoery for disributed matrices
template <typename TensorDataType, data_layout T_layout, El: Device Dev>
void graph_convolution_layer <TensorDataType, T_layout, Dev> 
::setup_matrices(const El::Grid& grid) {
  deallocate_matrices();
  if(Dev == El::Device::CPU){
    if (T_layout ==data_layout::MODEL_PARALLEL){
    } else if (T_layout == data_layout::DATA_PARALLEL){
      this->m_bias_gradient = 
        new El:DistMatrix<TensorDataType,
                          El::MC,El::STAR,
                          El::ELEMENT,
                          El::Device::CPU>(grid);
    } else if (T_layout == data_layout::DATA_PARALLEL){
      this->m_bias_gradient =
        new El::DistMatrix<TensorDataType,
                           El::STAR,El::STAR,
                           El::ELEMENT,
                           El::Device::CPU>(grid);
    }
  }
} // Setup Matrices 
template <typename TensorDataType, data_layout T_layout, El: Device Dev>
graph_convolution_layer * graph_convolution_layer <TensorDataType, T_layout, Dev> 
:: copy (){
} // Copy 
template <typename TensorDataType, data_layout T_layout, El: Device Dev>
void graph_convolution_layer <TensorDataType, T_layout, Dev> 
setup_data(size_t max_mini_batch_size){
  const auto& node_features = 

} // Setup Data 

template <typename TensorDataType>
void fp_compute_impl(graph_convolution_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::CPU>& l){
  // Matrices 

  const auto& local_node_features = l.get_local_prev_activations(0);
  const auto& local_adjacency_mat = l.get_local_prev_activations(1);
  
  get_num_parents
 } //Data_Parallel CPU Forward Prop

template <typename TensorDataType>
void bp_compute_impl(graph_convolution_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l){
  // Matrices 
  const auto& local_node_features = l.get_local_prev_activations(0);
  const auto& local_adjacency_mat = l.get_local_prev_activations(1); 

  //Use the identity (AB).T = (B.T)(A.T)

} // Data_Parellel CPU Bavkward Prop 

} //namespace lbann

