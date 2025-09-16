#pragma once
/**
 * @file SpikingNeuralNetwork.hpp
 * @brief SNN全体の初期化・接続生成・実行・出力の宣言
 */

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <unordered_map>

#include "Neuron.hpp"

class SpikingNeuralNetwork {
protected:
  std::vector<std::vector<std::shared_ptr<Neuron>>> neurons;   // [layer][i]
  std::vector<std::vector<std::shared_ptr<Synapse>>> synapses; // [pre_layer][k]
  std::mt19937 rng_;
  uint32_t seed_;

  int current_step = 0;

public:
  explicit SpikingNeuralNetwork(uint32_t seed = 0);

  // ----- 初期化／設定読み込み -----
  void init(const std::string& filename);

  // ----- 実行 -----
  void set_inputs(const std::vector<double>& values);
  void step();
  void run(int steps);

  // ----- STDP Freeze 制御 -----
  void disableSTDP(int layer_id, int neuron_id);
  void disableSTDP(int layer_id);
  void enableSTDP(int layer_id, int neuron_id);
  void enableSTDP(int layer_id);

  // ----- Sleep モード切替（追加） -----
  void set_sleep_mode_all_spiking(bool on, double rate_hz = 0.0);
  void set_sleep_mode_on_layer(int layer_id, bool on, double rate_hz = 0.0);
  void set_sleep_mode_on_neuron(int layer_id, int neuron_id, bool on, double rate_hz = 0.0);

  // ----- 可視化用 -----
  void export_graph_bin(const std::string& filename) const;

  // ----- ★テスト用ゲッター（読み取り専用） -----
  const std::vector<std::vector<std::shared_ptr<Neuron>>>& get_neurons() const { return neurons; }
  const std::vector<std::vector<std::shared_ptr<Synapse>>>& get_synapses() const { return synapses; }

protected:
  // AMPA→GABA の安定ソート
  void sort_synapses_by_type();

  // 解析ヘルパ
  void parse_and_bucket_lines(const std::string& filename,
                              std::vector<std::vector<std::string>>& out_buckets);
  void build_template_map(const std::vector<std::string>& template_lines,
                          std::unordered_map<std::string, std::string>& template_map);
  void apply_group_add(const std::vector<std::string>& group_lines,
                       const std::unordered_map<std::string, std::string>& template_map,
                       std::unordered_map<std::string, int>& layer_name_to_id);
  void apply_edges(const std::vector<std::string>& edge_lines,
                   const std::unordered_map<std::string, std::string>& template_map,
                   const std::unordered_map<std::string, int>& layer_name_to_id);

  // テンプレート生成
  template <class NeuronT, class... Args>
  void add_neurons(int layer_id, int count, Args&&... args) {
    if (static_cast<int>(neurons.size()) <= layer_id) neurons.resize(layer_id + 1);
    for (int i = 0; i < count; ++i) {
      neurons[layer_id].push_back(std::make_shared<NeuronT>(
          layer_id, static_cast<int>(neurons[layer_id].size()),
          std::forward<Args>(args)...));
    }
  }

  template <class SynT>
  void add_synapse(int pre_layer, int pre_id, int post_layer, int post_id,
                   double w, int flush_interval_sec, double dt,
                   double A_plus, double A_minus, double tau_plus, double tau_minus) {
    if (pre_layer < 0 || pre_layer >= static_cast<int>(synapses.size())) return;
    if (pre_id < 0 || pre_id >= static_cast<int>(neurons[pre_layer].size())) return;
    if (post_layer < 0 || post_layer >= static_cast<int>(neurons.size())) return;
    if (post_id < 0 || post_id >= static_cast<int>(neurons[post_layer].size())) return;
    synapses[pre_layer].push_back(std::make_shared<SynT>(
        neurons[pre_layer][pre_id], neurons[post_layer][post_id],
        w, flush_interval_sec, dt, A_plus, A_minus, tau_plus, tau_minus));
  }

  template <class SynT>
  bool remove_synapse_if_exists(int pre_layer, int pre_id, int post_layer, int post_id) {
    if (pre_layer < 0 || pre_layer >= static_cast<int>(synapses.size())) return false;
    auto& vec = synapses[pre_layer];
    const size_t before = vec.size();
    vec.erase(std::remove_if(vec.begin(), vec.end(),
              [&](const std::shared_ptr<Synapse>& s){
                auto pre = s->get_pre_neuron();
                auto post= s->get_post_neuron();
                return pre->get_layer_id()==pre_layer && pre->get_neuron_id()==pre_id &&
                       post->get_layer_id()==post_layer && post->get_neuron_id()==post_id &&
                       ( (s->type_code()==0 && std::is_same<SynT,ExcitatorySynapse>::value) ||
                         (s->type_code()==1 && std::is_same<SynT,InhibitorySynapse>::value) );
              }), vec.end());
    return vec.size() != before;
  }

  template <class SynT>
  void add_full_connect(int pre_layer, int post_layer, double w, int flush_interval_sec, double dt,
                        double A_plus, double A_minus, double tau_plus, double tau_minus) {
    const int n_pre  = static_cast<int>(neurons[pre_layer].size());
    const int n_post = static_cast<int>(neurons[post_layer].size());
    for (int i = 0; i < n_pre; ++i) {
      for (int j = 0; j < n_post; ++j) {
        add_synapse<SynT>(pre_layer, i, post_layer, j, w, flush_interval_sec, dt,
                          A_plus, A_minus, tau_plus, tau_minus);
      }
    }
  }

  template <class SynT>
  void add_random_connect(int pre_layer, int post_layer, int n_per_post,
                          double w, int flush_interval_sec, double dt,
                          double A_plus, double A_minus, double tau_plus, double tau_minus) {
    const int n_pre  = static_cast<int>(neurons[pre_layer].size());
    const int n_post = static_cast<int>(neurons[post_layer].size());
    if (n_pre <= 0 || n_post <= 0) return;

    std::vector<int> idx(n_pre);
    for (int j = 0; j < n_post; ++j) {
      std::iota(idx.begin(), idx.end(), 0);
      std::shuffle(idx.begin(), idx.end(), rng_);
      const int k = std::min(n_per_post, n_pre);
      for (int t = 0; t < k; ++t) {
        add_synapse<SynT>(pre_layer, idx[t], post_layer, j, w, flush_interval_sec, dt,
                          A_plus, A_minus, tau_plus, tau_minus);
      }
    }
  }
};
