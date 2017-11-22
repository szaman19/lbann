////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
// lbann_model .hpp .cpp - Abstract class for neural network training models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_HPP
#define LBANN_MODEL_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/weights/weights.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace lbann {

// Forward-declare this.
class lbann_callback;

/** Base class for LBANN models. */
class model {
 public:
  model(lbann_comm *comm,
        int mini_batch_size,
        objective_function *obj_fn,
        optimizer* default_optimizer = nullptr);
  model(const model& other);
  model& operator=(const model& other);
  virtual ~model();

  virtual model* copy() const = 0;

  /** Return the model's name. */
  virtual std::string name() const = 0;

  /** Initialize the model. */
  virtual void setup() {}

  /** Add layer to model. */
  virtual void add_layer(Layer *layer);

  /** Add weights to model. */
  void add_weights(weights *w);

  /** Register a new callback for the model. */
  void add_callback(lbann_callback *cb);

  /** Register a new metric for the model. */
  void add_metric(metrics::metric *m);

  /** Construct an instance of the default optimizer.
   *  If there is no default optimizer, a null pointer is returned.
   */
  optimizer* create_optimizer() const;

  /** Return the model's objective function. */
  objective_function* get_objective_function() {
    return m_objective_function;
  }

  /** Return the model's metrics. */
  virtual std::vector<metrics::metric *>& get_metrics() {
    return m_metrics;
  }

  /** Set the model's layers. */
  void set_layers(std::vector<Layer *>& layers);

  /** Return the model's layers. */
  std::vector<Layer *>& get_layers() { return m_layers; }

  /** Set the model's weights. */
  void set_weights(std::vector<weights *>& w);

  /** Return the model's weights. */
  std::vector<weights *>& get_weights() { return m_weights; }

  /** Get the model's comm. */
  inline lbann_comm *get_comm() const {
    return m_comm;
  }
  /** Get the current epoch for the model. */
  inline int get_cur_epoch() const {
    return m_current_epoch;
  }
  /** Get the current step for the model. */
  inline int get_cur_step() const {
    return m_current_step;  /// @todo This should be renamed to get_cur_training step and replaced with one that returns the current based on execution mode
  }
  
  /** Get the current validation step for the model. */
  inline int get_cur_validation_step() const {
    return m_current_validation_step;
  }
  /** Get the current testing step for the model. */
  inline int get_cur_testing_step() const {
    return m_current_testing_step;
  }
  /** Set the model (and all layers') execution mode. */
  virtual void set_execution_mode(execution_mode mode);
  /** Get the model's execution mode. */
  inline execution_mode get_execution_mode() const {
    return m_execution_mode;
  }
  /** Set the model's current mini-batch size. */
  inline void set_current_mini_batch_size(int mini_batch_size) {
    m_current_mini_batch_size = mini_batch_size;
  }
  /** Get the model's current mini-batch size. */
  inline int get_current_mini_batch_size() const {
    return m_current_mini_batch_size;
  }
  /** Get the model's maximum mini-batch size. */
  inline int get_max_mini_batch_size() const {
    return m_max_mini_batch_size;
  }
  /** Get the model's effective mini-batch size. */
  inline int get_effective_mini_batch_size() const {
    return m_effective_mini_batch_size;
  }
  /** Set the model's effective mini-batch size. */
  inline void set_effective_mini_batch_size(int mini_batch_size) {
    m_effective_mini_batch_size = mini_batch_size;
  }

  /** Get the current phase (multiple epochs) in layer-wise model training. */
  inline int get_current_phase() {
    return m_current_phase;
  }

  /**
   * Summarize statistics (e.g. timers, counters); these should be computable
   * quickly.
   */
  virtual void summarize_stats(lbann_summary& summarizer);
  /**
   * Summarize matrices (e.g. means); these are called less frequently and can
   * be more expensive.
   */
  virtual void summarize_matrices(lbann_summary& summarizer);

  /** Return true if the flag to stop training is set. */
  bool get_terminate_training() const {
    return m_terminate_training;
  }
  /** Set the terminate training flag (on or off). */
  void set_terminate_training(bool f) {
    m_terminate_training = f;
  }

  /** Train model. */
  virtual void train(int num_epochs);
  /** Evaluate model. */
  virtual void evaluate(execution_mode mode);

  /** Set checkpoint values */
  inline void set_checkpoint_dir(std::string dir)   {
    m_checkpoint_dir    = dir;
  }
  inline void set_checkpoint_epochs(int epochs) {
    m_checkpoint_epochs = epochs;
  }
  inline void set_checkpoint_steps(int steps)   {
    m_checkpoint_steps  = steps;
  }
  inline void set_checkpoint_secs(double secs)      {
    m_checkpoint_secs   = secs;
  }

#if 0
  /** Return true if about to start a new training epoch
   */
  virtual bool at_epoch_start() {
    return true;
  }

  /** Returns true if a checkpoint should be taken, false otherwise */
  bool need_checkpoint();

  /** Checkpoint model to given file descriptor, return number of bytes written */
  virtual bool save_to_checkpoint_shared(persist& p);
  /** Restore model by reading checkpoint from given file descriptor, return number of bytes read */
  virtual bool load_from_checkpoint_shared(persist& p);

  /*! Top-level call to start checkpoint.  This creates the persist object
   *  and then calls the model's save_to_checkpoint_shared() virtual function */
  bool checkpointShared();

  /*! Top-level call to restart.  This creates the persist object
   *  and then calls the model's load_from_checkpoint_shared() virtual function */
  bool restartShared();

#endif // 0

 protected:

  /** The objective function used to train the model. */
  objective_function *m_objective_function;

  /** The model's current execution mode. */
  execution_mode m_execution_mode;
  /** Flag telling the model to terminate training. */
  bool m_terminate_training;
  /** Most recent/current epoch for the model. */
  int m_current_epoch;
  /** Most recent/current training step for the model. */
  int m_current_step;
  int m_current_validation_step;
  int m_current_testing_step;
  /**
   * Maximum possible minibatch size supported by layers in this model.
   * Note that this is local to the particular model, not across multiple
   * models.
   */
  int m_max_mini_batch_size;
  /** Size of the current mini-batch in the model. */
  int m_current_mini_batch_size;
  /**
   * The "effective" size of a minibatch.
   * This is the size of the minibatch across all models and used for e.g.
   * correctly averaging gradients from multiple models.
   */
  int m_effective_mini_batch_size;
  /** current phase (multiple of epoch counts) in training a model */
  int m_current_phase;
  /** Communicator for the model. */
  lbann_comm *m_comm;
  /** Current callbacks to process. */
  std::vector<lbann_callback *> m_callbacks;

  /** Directory where we should save checkpoints */
  std::string m_checkpoint_dir;
  /** Number of training steps to elapse between checkpoints */
  int m_checkpoint_epochs;
  /** Number of training steps to elapse between checkpoints */
  int m_checkpoint_steps;
  /** Number of seconds to elapse between checkpoints (checkpoint interval) */
  double m_checkpoint_secs;
  /** Timestamp of last checkpoint */
  double m_checkpoint_last;

  optimizer *m_default_optimizer;

  /**
   * A metric is a function that is used to judge the performance of your model.
   * A metric function is similar to an objective function, except that the
   * results from evaluating a metric are not used when training the model.
   */
  std::vector<metrics::metric *> m_metrics;

  /** List of layers in model.
   *  The list is in execution order for forward propagation.
   */
  std::vector<Layer *> m_layers;

  std::vector<weights *> m_weights;

  /** Check if the model (and all layers') execution mode valid. */
  virtual bool is_execution_mode_valid(execution_mode mode) const;
  /// Print out the description of a layer set up
  virtual std::string print_layer_description(const Layer* layer) const;

  /// Deallocate layer objects
  virtual void forward_prop_to_evaluate();
  virtual void forward_prop();
  virtual void backward_prop();
  /** Train model on a mini-batch. */
  virtual bool train_mini_batch();
  /** Evaluate model on a mini-batch */
  virtual bool evaluate_mini_batch();

  // Methods for calling every callback at different points.
  virtual void setup_callbacks();
  virtual void do_train_begin_cbs();
  virtual void do_train_end_cbs();
  virtual void do_phase_end_cbs();
  virtual void do_epoch_begin_cbs();
  virtual void do_epoch_end_cbs();
  virtual void do_batch_begin_cbs();
  virtual void do_batch_end_cbs();
  virtual void do_test_begin_cbs();
  virtual void do_test_end_cbs();
  virtual void do_validation_begin_cbs();
  virtual void do_validation_end_cbs();
  virtual void do_model_forward_prop_begin_cbs();
  virtual void do_layer_forward_prop_begin_cbs(Layer *l);
  virtual void do_model_forward_prop_end_cbs();
  virtual void do_layer_forward_prop_end_cbs(Layer *l);
  virtual void do_model_backward_prop_begin_cbs();
  virtual void do_layer_backward_prop_begin_cbs(Layer *l);
  virtual void do_model_backward_prop_end_cbs();
  virtual void do_layer_backward_prop_end_cbs(Layer *l);
  /// Evaluation phases (validation / testing)
  virtual void do_batch_evaluate_begin_cbs();
  virtual void do_batch_evaluate_end_cbs();
  virtual void do_model_evaluate_forward_prop_begin_cbs();
  virtual void do_layer_evaluate_forward_prop_begin_cbs(Layer *l);
  virtual void do_model_evaluate_forward_prop_end_cbs();
  virtual void do_layer_evaluate_forward_prop_end_cbs(Layer *l);
};

}  // namespace lbann

#endif  // LBANN_MODEL_HPP
