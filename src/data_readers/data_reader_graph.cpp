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

#include "lbann/data_readers/data_reader_graph.hpp"
#include "lbann/trainers/trainer.hpp"
#include <iostream>

namespace lbann{

data_reader_graph::data_reader_graph(int max_node_size,
                                     int max_edge_size,
                                     int num_node_features,
                                     int num_edge_features,
                                     bool has_edge_features,
                                     bool shuffle)
  : generic_data_reader(shuffle),
    m_max_node_size(max_node_size),
    m_max_edge_size(max_edge_size),
    m_num_node_features(num_node_features),
    m_num_edge_features(num_edge_features),
    m_has_edge_features(has_edge_features){
    if (has_edge_features && num_edge_features == 0)
    {
      LBANN_ERROR("Expected edge features but number of edge features no specified");
    }

    if (!has_edge_features && num_edge_features > 0){
      LBANN_ERROR("Expected no edge features but number of edge features is greater than 0");
    } 

  }

bool data_reader_graph::fetch_datum(CPUMat& X, int data_id, int mb_idx){
  size_t j; 

  // Fetch node fetures

  auto num_nodes = m_num_nodes[data_id];
  auto num_edges = m_num_edges[data_id];


  for(j = 0; j < num_nodes; ++j){
    for (auto i = 0; i < m_num_node_features; ++i){
      X(j*m_num_node_features+i, mb_idx) = m_node_features[j][i];
    }
  }

  // Pad the rest of the node features with 0

  for (; j < m_max_node_size * m_num_node_features; ++j){
    for (auto i = 0; i < m_num_node_features; ++i){
      X(j*m_num_node_features+i, mb_idx) = 0;  
    }
  }

  // If graph has edge features, fetch edge features

  auto feature_end = m_num_node_features * m_num_nodes;

  if (m_has_edge_features){
    size_t edge; 
    for (edge = 0; edge < num_edges; ++edge){
      for(auto i = 0; i < m_num_edge_features; ++i){
        X(feature_end+edge*m_num_edge_features+i, mb_idx) = m_edge_features[edge][i];
      }
    }

    for(; edge < m_max_edge_size; ++edge){
      for(auto i = 0; i < m_num_edge_features; ++i){
        X(feature_end+edge*m_num_edge_features+i, mb_idx) = 0;        
      }

    feature_end = feature_end + m_num_edge_features * num_edges;
    }
  }

  // Fetch Source Node Indices

  size_t ind;
  for(ind = 0; ind < num_edges; ++ind){
    X(feature_end+ind, mb_idx) = m_source_nodes[ind];
  }

  // Pad the rest of the indices with -1
  for(; ind<m_max_edge_size; ++ind){
    X(feature_end+ind, mb_idx) = -1;
  }

  feature_end = feature_end + m_max_edge_size;

  // Fetch Target Node Indices

  for(ind = 0; ind < num_edges; ++ind){
    X(feature_end+ind, mb_idx) = m_target_nodes[ind];
  }

  // Pad the rest of the indices with -1
  for(; ind<m_max_edge_size; ++ind){
    X(feature_end+ind, mb_idx) = -1;
  }

  feature_end = feature_end + m_max_edge_size;

  // Load the target value for the graph
  X(feature_end+1, mb_idx) = m_labels[data_id];


  return true;
}

bool data_reader_graph::fetch_response(CPUMat& Y, int data_id, int mb_idx){
  return true;
}

bool data_reader_graph::fetch_label(CPUMat& Y, int data_id, int mb_idx){
  return true;
}

void data_reader_graph::load_graph_data(const std::string input_filename){
  // TO DO: Load the entire dataset into appropriate vectors

  std::ifsteam in(input_filename.c_str(), std::ios::binary);

  if(!in){
    LBANN_ERROR("failed to open data file: ", fn, '\n');
  }

  std::cout << "Data file succesfully opened \n";



  while(sample < m_num_samples){
    int num_nodes;
    int num_edges;

    in.read(reinterpret_cast<char*>(&num_nodes), sizeof(int));
    in.read(reinterpret_cast<char*>(&num_edges), sizeof(int));

    std::vector<float> node_features (num_nodes);
    in.read(reinterpret_cast<char *>(&node_features[0]), num_nodes*m_num_node_features*sizeof(float));

    if (m_has_edge_features){
      vector<float> edge_features (num_edges);
      in.read(reinterpret_cast<char *>(&edge_features[0]), num_edges*m_num_edge_features*sizeof(float));
      m_edge_features.append(edge_features);
    }

    std::vector<int> source_nodes; 
    std::vector<int> target_nodes; 

    in.read(reinterpret_cast<char *>(&source_nodes[0]), num_edges*sizeof(int));
    in.read(reinterpret_cast<char *>(&target_nodes[0]), num_edges*sizeof(int));


    ++sample;
    m_num_nodes.append(num_nodes);
    m_num_edges.append(num_edges);
    m_node_features.append(node_features);
    m_source_nodes.append(source_nodes);
    m_target_nodes.append(target_nodes);

  }

}

void data_reader_graph::load(){
  if (is_master()){
    std::cout <, "Starting lbann::graph_reader::load \n";
  }
  m_num_nodes.clear();
  m_num_edges.clear();
  m_node_features.clear();
  m_edge_features.clear();
  m_source_nodes.clear();
  m_target_nodes.clear();
  m_labels.clear();

}

} // namespace lbann
