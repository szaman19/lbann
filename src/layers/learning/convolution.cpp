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

#define LBANN_CONVOLUTION_LAYER_INSTANTIATE
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/learning/convolution.hpp"

#include "lbann/proto/proto_common.hpp"

#include <layers.pb.h>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
convolution_layer<TensorDataType,Layout,Device>::convolution_layer(
  lbann_comm *comm,
  int num_data_dims,
  int num_output_channels,
  int conv_dim,
  int pad,
  int stride,
  int dilation,
  int groups,
  bool has_bias)
  : convolution_layer(comm,
                      num_data_dims,
                      num_output_channels,
                      std::vector<int>(num_data_dims, conv_dim),
                      std::vector<int>(num_data_dims, pad),
                      std::vector<int>(num_data_dims, stride),
                      std::vector<int>(num_data_dims, dilation),
                      groups,
                      has_bias)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
convolution_layer<TensorDataType,Layout,Device>::convolution_layer(
  lbann_comm *comm,
  int num_data_dims,
  int num_output_channels,
  std::vector<int> conv_dims,
  std::vector<int> pads,
  std::vector<int> strides,
  std::vector<int> dilations,
  int groups,
  bool has_bias)
  : base_convolution_layer<TensorDataType, Device>(
    comm,
    num_data_dims,
    num_output_channels,
    std::move(conv_dims),
    std::move(pads),
    std::move(strides),
    std::move(dilations),
    groups,
    has_bias)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType,Layout,Device>
::setup_dims(DataReaderMetaData& dr_metadata)  {
  base_convolution_layer<TensorDataType, Device>::setup_dims(dr_metadata);

  // Get tensor dimensions
  const auto& input_dims = this->get_input_dims();
  auto output_dims = input_dims;

  // Initialize output tensor dimensions
  output_dims[0] = this->m_output_channels;
  for (size_t i = 0; i < output_dims.size() - 1; ++i) {
    const auto& input_dim = input_dims[i+1];
    const auto& kernel_dim = this->m_conv_dims[i];
    const auto& stride = this->m_strides[i];
    const auto& pad = this->m_pads[i];
    const auto& dilation = this->m_dilations[i];
    const auto& effective_dim = (input_dim
                                 + 2 * pad
                                 - dilation * (kernel_dim-1));
    output_dims[i+1] = (effective_dim + stride - 1) / stride;
  }
  this->set_output_dims(output_dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::vector<int> convolution_layer<TensorDataType,Layout,Device>
::get_kernel_dims() const {
  std::vector<int> dims;
  dims.push_back(this->m_output_channels);
  dims.push_back(this->get_input_dims()[0] / this->m_groups);
  dims.insert(dims.end(),
              this->m_conv_dims.begin(),
              this->m_conv_dims.end());
  return dims;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType,Layout,Device>::fp_compute() {
  using BaseConvLayer = base_convolution_layer<TensorDataType, Device>;
  if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      this->get_distconv_adapter().fp_compute_convolution();
      this->get_distconv_adapter().fp_apply_bias();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    BaseConvLayer::apply_convolution_cudnn(true);
    BaseConvLayer::apply_bias_cudnn();
  }
  else {
    BaseConvLayer::apply_convolution_im2col(true);
    BaseConvLayer::apply_bias_cpu();
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType,Layout,Device>::bp_compute() {
  using BaseConvLayer = base_convolution_layer<TensorDataType, Device>;
  if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      if (this->get_distconv_adapter().m_conv->is_overlap_bwd_halo_exchange_enabled()) {
        this->get_distconv_adapter().m_conv->backward_data_exchange_halo(
          this->get_distconv_adapter().get_prev_error_signals());
      }
      this->get_distconv_adapter().bp_compute_convolution_filter();
      this->get_distconv_adapter().bp_compute_convolution_data();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    BaseConvLayer::compute_gradients_cudnn(false);
    BaseConvLayer::apply_transposed_convolution_cudnn(false);
  }
  else {
    BaseConvLayer::compute_gradients_im2col(false);
    BaseConvLayer::apply_transposed_convolution_im2col(false);
  }
}

#if defined LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_layer<TensorDataType,Layout,Device>::setup_distconv_adapter() {
  this->get_distconv_adapter_ptr() = make_unique<
    convolution_distconv_adapter<TensorDataType, Layout, Device>>(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool convolution_layer<TensorDataType,Layout,Device>
::is_distconv_supported() const {
  const auto& kernel_dims = get_kernel_dims();
  for(int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
    if (kernel_dims[2 + i] != kernel_dims[2]) {
      dc::MPIRootPrintStreamDebug()
        << "Nonsymmetric kernel not supported";
      return false;
    }
    if (kernel_dims[2 + i] !=
        this->m_pads[i] / this->m_dilations[i] * 2 + 1) {
      dc::MPIRootPrintStreamDebug()
        << "Unsupported as padding does not match the kernel size";
      return false;
    }
  }
  return true;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void convolution_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(tensor_overlap_constraints &constraints) {
  base_convolution_adapter<TensorDataType, Dev>::setup_distributions(
      constraints);
  auto &l = dynamic_cast<convolution_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());
  auto kernel_dims = l.get_kernel_dims();
  std::reverse(kernel_dims.begin(), kernel_dims.end());
  auto dilations = l.m_dilations;
  std::reverse(dilations.begin(), dilations.end());
  dc::IntVector overlap(dc::get_num_dims(l), 0);
  const auto &ps = l.get_parallel_strategy();
  // i=0 -> width; i=1 -> height; i=2: -> depth;
  for(int i = 0; i < dc::get_num_spatial_dims(l); i++) {
    int splits = 0;
    switch (i) {
      case 0: splits = ps.width_splits; break;
      case 1: splits = ps.height_splits; break;
      case 2: splits = ps.depth_splits; break;
    }
    if (splits > 1) {
      overlap[i] = (kernel_dims[i] - 1) / 2 * dilations[i];
    }
  }
  auto &prev_activations_dist = this->get_prev_activations_dist();
  prev_activations_dist.set_overlap(overlap);
  constraints.mark_updated(prev_activations_dist);
  constraints.mark_invariant(prev_activations_dist);
  auto &prev_error_signals_dist = this->get_prev_error_signals_dist();
  prev_error_signals_dist.set_overlap(overlap);
  constraints.mark_updated(prev_error_signals_dist);
  constraints.mark_invariant(prev_error_signals_dist);
  // To deal with strides, error signals must have the same size
  // of overlap
  auto &error_signals_dist = this->get_error_signals_dist();
  error_signals_dist.set_overlap(overlap);
  constraints.mark_updated(error_signals_dist);
  constraints.mark_invariant(error_signals_dist);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape convolution_distconv_adapter<TensorDataType, Layout, Device>::
get_activations_local_shape(int index) const {
  assert_eq(index, 0);
  const auto &layer = dynamic_cast<const convolution_layer<
    TensorDataType, Layout, Device>&>(this->layer());
  auto filter_dims = layer.get_kernel_dims();
  std::reverse(std::begin(filter_dims), std::end(filter_dims));
  auto strides = layer.m_strides;
  std::reverse(std::begin(strides), std::end(strides));
  auto dilations = layer.m_dilations;
  std::reverse(std::begin(dilations), std::end(dilations));
  const auto output_spatial_local_shape =
      ::distconv::get_convolution_output_local_tensor_shape(
          this->get_prev_activations(),
          filter_dims, strides, true, dilations,
          layer.m_groups);
  return output_spatial_local_shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
    size_t workspace_capacity) {
  base_convolution_adapter<TensorDataType, Device>::setup_layer(
      workspace_capacity);
  auto &layer = dynamic_cast<convolution_layer<
    TensorDataType, Layout, Device>&>(this->layer());

  if (dc::is_deterministic()) {
    dc::MPIRootPrintStreamDebug()
      << "Using deterministic convolution algorithms";
    this->m_fwd_algo = "DETERMINISTIC";
    this->m_bwd_data_algo = "DETERMINISTIC";
    this->m_bwd_filter_algo = "DETERMINISTIC";
  } else {
    this->m_fwd_algo = dc::get_convolution_fwd_algorithm();
    this->m_bwd_data_algo = dc::get_convolution_bwd_data_algorithm();
    this->m_bwd_filter_algo = dc::get_convolution_bwd_filter_algorithm();
  }

  std::vector<int> pads = layer.m_pads;
  std::reverse(pads.begin(), pads.end());
  std::vector<int> strides = layer.m_strides;
  std::reverse(strides.begin(), strides.end());
  std::vector<int> dilations = layer.m_dilations;
  std::reverse(dilations.begin(), dilations.end());

  this->m_conv->setup(this->get_prev_activations(),
                      *(this->m_kernel), this->get_activations(),
                      this->get_error_signals(),
                      *this->m_kernel_gradient,
                      this->get_prev_error_signals(),
                      pads, strides, dilations, layer.m_groups,
                      this->m_fwd_algo, this->m_bwd_data_algo,
                      this->m_bwd_filter_algo,
                      workspace_capacity);
}
#endif // defined LBANN_HAS_DISTCONV

// Builder helper stuff
namespace {

#ifdef LBANN_HAS_CUDNN
using ProtoTensorOpEnumType = decltype(lbann_data::DEFAULT_TENSOR_OPS);
cudnnMathType_t convert_to_cudnn_math_type(ProtoTensorOpEnumType mt)
{
  switch (mt)
  {
  case lbann_data::DEFAULT_TENSOR_OPS:
    return cudnn::get_default_convolution_math_type();
  case lbann_data::NO_TENSOR_OPS:
    return CUDNN_DEFAULT_MATH;
  case lbann_data::USE_TENSOR_OPS:
    return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  default:
    LBANN_ERROR("Bad math type value.");
  }
  return CUDNN_DEFAULT_MATH;
}
#endif // LBANN_HAS_CUDNN

template <typename TensorDataType, data_layout Layout, El::Device Device>
struct ConvLayerBuilder
{
  static std::unique_ptr<Layer> Build(
    lbann_comm* comm, lbann_data::Layer const& proto_layer){

    const auto& params = proto_layer.convolution();
    const auto& num_output_channels = params.num_output_channels();
    const auto& bias = params.has_bias();
    int num_groups = params.num_groups();
    if (num_groups == 0) {
      num_groups = 1;
    }

    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      std::vector<int> dilations = parse_list<int>(params.conv_dilations());
      if (dilations.empty()) {
        dilations.resize(dims.size(), 1);
      }
#ifdef LBANN_HAS_CUDNN
      auto ret = lbann::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        comm, dims.size(), num_output_channels,
        dims, pads, strides, dilations, num_groups, bias);
      ret->set_cudnn_math_mode(
        convert_to_cudnn_math_type(params.conv_tensor_op_mode()));
      return ret;
#else
      return lbann::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        comm, dims.size(), num_output_channels,
        dims, pads, strides, dilations, num_groups, bias);
#endif // LBANN_HAS_CUDNN
    }
    else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      int dilation = params.conv_dilations_i();
      if (dilation == 0) {
        dilation = 1;
      }
#ifdef LBANN_HAS_CUDNN
      auto ret =lbann::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        comm, num_dims, num_output_channels,
        dim, pad, stride, dilation, num_groups, bias);
      ret->set_cudnn_math_mode(
        convert_to_cudnn_math_type(params.conv_tensor_op_mode()));
      return ret;
#else
      return lbann::make_unique<convolution_layer<TensorDataType, Layout, Device>>(
        comm, num_dims, num_output_channels,
        dim, pad, stride, dilation, num_groups, bias);
#endif // LBANN_HAS_CUDNN
    }
  }
};

template <typename TensorDataType, El::Device Device>
struct ConvLayerBuilder<TensorDataType, data_layout::MODEL_PARALLEL, Device>
{
  static std::unique_ptr<Layer> Build(
    lbann_comm* comm, lbann_data::Layer const& proto_layer){
    LBANN_ERROR("convolution layer is only supported with "
                "a data-parallel layout");
  }
};

}// namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_convolution_layer_from_pbuf(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer) {
  using Builder = ConvLayerBuilder<TensorDataType, Layout, Device>;
  return Builder::Build(comm, proto_layer);
}

#define PROTO_DEVICE(T, Device)                                            \
  template class convolution_layer<T, data_layout::DATA_PARALLEL, Device>; \
    template std::unique_ptr<Layer>                                       \
  build_convolution_layer_from_pbuf<T, data_layout::DATA_PARALLEL, Device>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_convolution_layer_from_pbuf<T, data_layout::MODEL_PARALLEL, Device>( \
    lbann_comm*, lbann_data::Layer const&)

#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
