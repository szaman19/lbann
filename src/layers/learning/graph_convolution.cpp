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
graph_convolution_layer <TensorDataType, T_layout, Dev> :: graph_convolution_layer(
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


} // Constructor 

} //namespace lbann

