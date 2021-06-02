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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_GRAPH_HPP
#define LBANN_DATA_READER_GRAPH_HPP

#include "data_reader.hpp"

namespace lbann{
  /**
  * Data reader for  parsing graph structured data
  *
  */
  class data_reader_graph : public generic_data_reader {
  public:
    data_reader_graph(int max_node_size,
                      int max_edge_size,
                      int num_node_features,
                      int num_edge_features = 0,
                      bool has_edge_features = false,
                      bool shuffle = true);
    data_reader_graph(const data_reader_graph&) = default;
    data_reader_graph& operator=(const data_reader_graph&) = default;
    ~data_reader_graph() override;
    data_reader_graph* copy() const override (return new data_reader_graph(*this));

    std::string get_type() const override{
      return "graph_reader";
    }
    void load() override;

    int get_linearized_data_size() const override{

    } 

  protected:
    bool fetch_datum(CPUMat& X, int data_id. int mb_idx) override;
    bool fetch_label(CPUMat& Y) override;
    bool fetch_response(CPUMat&) override;

  private:
    /** Number of samples in the dataset **/
    int m_num_samples;
    /** Number of nodes of graph with the most nodes in dataset (for padding) */
    int m_max_node_size; 
    /** Number of edges of graph with the most edges in dataset (for padding) */
    int m_max_edge_size; 
    /** Dimensionality of node features */
    int m_num_node_features;
    /** Dimensionality of edge features */
    int m_num_edge_features;
    /** Whether edge features exist */
    bool m_has_edge_features; 

    /**
     * Loaded graph data
     * 
     */
    std::vector<int> m_num_nodes;
    std::vector<int> m_num_edges;
    std::vector<std::vector<float>> m_node_features; 
    std::vector<std::vector<float>> m_edge_features;
    std::vector<std::vector<int>> m_source_nodes; 
    std::vecotr<std::vector<int>> m_target_nodes;  

    /** Loaded label information */
    std::vector<float> m_labels;

  }  // namespace lbann
}

#endif // LBANN_DATA_READER_GRAPH_HPP