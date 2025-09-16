/**
 * @file Neuron.cpp
 * @brief ニューロン／シナプス実装
 */

#include "Neuron.hpp"
#include <iostream>
#include <fstream>

// ------------------------------ Neuron ------------------------------
Neuron::Neuron(int layer_id, int neuron_id, double max_receptor,
               int flush_interval_sec, double dt)
    : layer_id_(layer_id),
      neuron_id_(neuron_id),
      flush_count_(0),
      id_str_(),
      input_current_(0.0),
      input_current_log_(0.0),
      membrane_potential_(0.0),
      receptor_pool_(max_receptor),
      max_receptor_(max_receptor),
      spike_history_(),
      membrane_potential_history_(),
      receptor_pool_history_(),
      input_current_history_(),
      record_history_(true),
      history_interval_steps_(static_cast<int>(flush_interval_sec / dt)),
      last_flush_steps_(-1),
      flush_interval_sec_(flush_interval_sec),
      force_spike_flag_(false),
      sum_weight_(0.0),
      connection_count_(0),
      header_str_(),
      freeze_(false) {
  std::filesystem::create_directories("./spiking_results");
  id_str_ = std::to_string(layer_id_) + "_" + std::to_string(neuron_id_);

  clear_current();
  clear_sum_weight();

  json header = {{"layer_id", layer_id_}, {"neuron_id", neuron_id_}, {"type", "NULL"}};
  header_str_ = header.dump();

  clear_history();
}

Neuron::~Neuron() {
  if (!spike_history_.empty()) {
    std::ostringstream oss;
    oss << "./spiking_results/" << id_str_ << "_final.bin";
    save_history_to_bin(oss.str());
  }
}

void Neuron::reset() {
  membrane_potential_ = 0.0;
  receptor_pool_ = max_receptor_;
  clear_history();
  last_flush_steps_ = -1;
  force_spike_flag_ = false;
  flush_count_ = 0;
  clear_current();   // 力学/ログともに0化
  clear_sum_weight();
}

void Neuron::restore_receptors(double amount) {
  receptor_pool_ += amount;
  if (receptor_pool_ > max_receptor_) receptor_pool_ = max_receptor_;
}

double Neuron::request_receptors(double amount) {
  const double actual = (receptor_pool_ > amount) ? amount : receptor_pool_;
  receptor_pool_ -= actual;
  return actual;
}

double Neuron::get_receptor_pool() const { return receptor_pool_; }

void Neuron::clear_history() {
  spike_history_.clear();
  membrane_potential_history_.clear();
  receptor_pool_history_.clear();
  input_current_history_.clear();
}

void Neuron::enable_history_recording(bool flag) { record_history_ = flag; }

void Neuron::record_current_state() {
  if (!record_history_) return;
  membrane_potential_history_.push_back(membrane_potential_);
  receptor_pool_history_.push_back(receptor_pool_);
  input_current_history_.push_back(input_current_log_); // ★ログ専用値を保存
}

void Neuron::check_and_save_history(int step) {
  if (!record_history_) return;
  if ((step - last_flush_steps_) >= history_interval_steps_) {
    std::ostringstream oss;
    oss << "./spiking_results/" << id_str_ << "_" << std::setw(4) << std::setfill('0')
        << flush_count_ << ".bin";
    save_history_to_bin(oss.str());
    clear_history();
    last_flush_steps_ = step;
    flush_count_++;
  }
}

void Neuron::save_history_to_bin(const std::string& filename) {
  const int32_t header_len = static_cast<int32_t>(header_str_.size());
  const int32_t n = static_cast<int32_t>(membrane_potential_history_.size());

  std::ofstream ofs(filename, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  ofs.write(header_str_.data(), header_len);
  ofs.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));

  std::vector<float> mp_float(membrane_potential_history_.begin(),
                              membrane_potential_history_.end());
  std::vector<float> rp_float(receptor_pool_history_.begin(),
                              receptor_pool_history_.end());
  std::vector<float> ic_float(input_current_history_.begin(),
                              input_current_history_.end());

  ofs.write(reinterpret_cast<const char*>(mp_float.data()), n * sizeof(float));
  ofs.write(reinterpret_cast<const char*>(rp_float.data()), n * sizeof(float));
  ofs.write(reinterpret_cast<const char*>(ic_float.data()), n * sizeof(float));
  ofs.write(reinterpret_cast<const char*>(spike_history_.data()), n * sizeof(uint8_t));
  ofs.close();
}

// ------------------------------ InputNeuron ------------------------------
InputNeuron::InputNeuron(int layer_id, int neuron_id, double max_receptor,
                         int flush_interval_sec, double dt,
                         std::optional<uint32_t> seed_opt)
    : Neuron(layer_id, neuron_id, max_receptor, flush_interval_sec, dt),
      input_value_(0.0),
      input_rate_(0.0),
      dt_(dt),
      spiking_now_(false),
      seed_(0),
      rng_(),
      uniform01_(0.0, 1.0) {
  if (seed_opt.has_value()) {
    seed_ = seed_opt.value();
  } else {
    std::random_device rd;
    seed_ = rd();
  }
  rng_ = std::mt19937(seed_);

  id_str_ = std::to_string(layer_id_) + "_" + std::to_string(neuron_id_) + "_INP";

  json header = {{"layer_id", layer_id_}, {"neuron_id", neuron_id_}, {"type", "INP"},
                 {"dt", dt_}, {"seed", seed_}};
  header_str_ = header.dump();
}

void InputNeuron::set_input(double value) {
  input_value_ = value;
  input_rate_ = std::max(0.0, input_value_);
}

void InputNeuron::step(int t) {
  spiking_now_ = false;
  double p_spike = input_rate_ * dt_;
  if (p_spike > 1.0) p_spike = 1.0;
  if (uniform01_(rng_) < p_spike) spiking_now_ = true;

  spike_history_.push_back(spiking_now_ ? 1 : 0);
  record_current_state();
  check_and_save_history(t);

  clear_sum_weight();
}

bool InputNeuron::is_spiking() const { return spiking_now_; }

void InputNeuron::reset() {
  Neuron::reset();
  input_value_ = 0.0;
  input_rate_ = 0.0;
  spiking_now_ = false;
}

// ------------------------------ SpikingNeuron ------------------------------
SpikingNeuron::SpikingNeuron(int layer_id, int neuron_id, double max_receptor,
                             int flush_interval_sec, double tau_m, double v_rest,
                             double v_reset, double v_th, double refractory_period,
                             double dt)
    : Neuron(layer_id, neuron_id, max_receptor, flush_interval_sec, dt),
      tau_m_(tau_m),
      v_rest_(v_rest),
      v_reset_(v_reset),
      v_th_(v_th),
      refractory_period_(refractory_period),
      spiking_now_(false),
      refractory_time_left_(0.0),
      dt_(dt) {
  membrane_potential_ = v_rest_;

  id_str_ = std::to_string(layer_id_) + "_" + std::to_string(neuron_id_) + "_SPK";
  json header = {{"layer_id", layer_id_},
                 {"neuron_id", neuron_id_},
                 {"type", "SPK"},
                 {"dt", dt_},
                 {"tau_m", tau_m_},
                 {"v_rest", v_rest_},
                 {"v_reset", v_reset_},
                 {"v_th", v_th_},
                 {"refractory_period", refractory_period_}};
  header_str_ = header.dump();

  // 睡眠用 RNG
  std::random_device rd;
  sleep_seed_ = rd();
  sleep_rng_  = std::mt19937(sleep_seed_);
}

void SpikingNeuron::set_sleep_mode(bool on, double rate_hz) {
  sleep_mode_ = on;
  if (on) {
    sleep_rate_hz_ = std::max(0.0, rate_hz);
    set_freeze(true);  // post側STDPを停止
  } else {
    sleep_rate_hz_ = 0.0;
    set_freeze(false); // freeze解除
  }
}

void SpikingNeuron::step(int t) {
  spiking_now_ = false;

  // 睡眠中：力学入力は 0 のまま、ログだけにパルスを残す
  if (sleep_mode_) {
    input_current_ = 0.0; // 力学に効く入力はゼロ固定
    double p_spike = sleep_rate_hz_ * dt_;
    if (p_spike > 1.0) p_spike = 1.0;
    if (sleep_u01_(sleep_rng_) < p_spike) {
      add_current_log_only(v_th_ - v_rest_); // ★ログ専用の可視パルス
      force_spike_flag_ = true;              // 発火は強制
    }
  }

  if (refractory_time_left_ > 1e-12) {
    membrane_potential_ = v_reset_;
    refractory_time_left_ -= dt_;
    if (refractory_time_left_ < 0.0) refractory_time_left_ = 0.0;
  } else {
    const double dV =
        (-(membrane_potential_ - v_rest_) + input_current_) * (dt_ / tau_m_);
    membrane_potential_ += dV;

    if ((membrane_potential_ >= v_th_) || force_spike_flag_) {
      spiking_now_ = true;
      record_current_state();         // 発火直前の状態を記録
      membrane_potential_ = v_reset_; // リセット
      refractory_time_left_ = refractory_period_;
      force_spike_flag_ = false;
    }
  }

  spike_history_.push_back(spiking_now_ ? 1 : 0);
  if (!spiking_now_) record_current_state();
  check_and_save_history(t);

  clear_current();   // 力学/ログを両方ゼロ化
  clear_sum_weight();
}

bool SpikingNeuron::is_spiking() const { return spiking_now_; }

void SpikingNeuron::reset() {
  Neuron::reset();
  membrane_potential_ = v_rest_;
  refractory_time_left_ = 0.0;
  spiking_now_ = false;
}

// ------------------------------ OutputNeuron ------------------------------
OutputNeuron::OutputNeuron(int layer_id, int neuron_id, double max_receptor,
                           int flush_interval_sec, double tau_m, double v_rest,
                           double v_reset, double v_th, double refractory_period,
                           double dt)
    : SpikingNeuron(layer_id, neuron_id, max_receptor, flush_interval_sec, tau_m,
                    v_rest, v_reset, v_th, refractory_period, dt) {
  id_str_ = std::to_string(layer_id_) + "_" + std::to_string(neuron_id_) + "_OUT";
  json header = {{"layer_id", layer_id_},
                 {"neuron_id", neuron_id_},
                 {"type", "OUT"},
                 {"dt", dt_},
                 {"tau_m", tau_m_},
                 {"v_rest", v_rest_},
                 {"v_reset", v_reset_},
                 {"v_th", v_th_},
                 {"refractory_period", refractory_period_}};
  header_str_ = header.dump();
}

// ------------------------------ Synapse ------------------------------
Synapse::Synapse(std::shared_ptr<Neuron> pre, std::shared_ptr<Neuron> post, double w,
                 int flush_interval_sec, double dt,
                 double A_plus, double A_minus, double tau_plus, double tau_minus)
    : pre_neuron_(std::move(pre)),
      post_neuron_(std::move(post)),
      weight_(w),
      synapse_name_(),
      pre_spike_history_(),
      post_spike_history_(),
      weight_history_(),
      receptor_pool_history_(),
      history_interval_steps_(static_cast<int>(flush_interval_sec / dt)),
      last_flush_step_(-1),
      flush_count_(0),
      A_plus_(A_plus),
      A_minus_(A_minus),
      tau_plus_(tau_plus),
      tau_minus_(tau_minus),
      dt_(dt),
      pre_spike_times_(),
      post_spike_times_(),
      header_str_() {
  json header = {{"pre_layer_id", pre_neuron_->get_layer_id()},
                 {"pre_neuron_id", pre_neuron_->get_neuron_id()},
                 {"post_layer_id", post_neuron_->get_layer_id()},
                 {"post_neuron_id", post_neuron_->get_neuron_id()},
                 {"type", "NULL"},
                 {"initial_weight", w},
                 {"A_plus", A_plus_},
                 {"A_minus", A_minus_},
                 {"tau_plus", tau_plus_},
                 {"tau_minus", tau_minus_},
                 {"dt", dt_}};
  header_str_ = header.dump();
}

void Synapse::record_current_state() {
  pre_spike_history_.push_back(pre_neuron_->is_spiking() ? 1 : 0);
  post_spike_history_.push_back(post_neuron_->is_spiking() ? 1 : 0);
  weight_history_.push_back(static_cast<float>(weight_));
  receptor_pool_history_.push_back(
      static_cast<float>(post_neuron_->get_receptor_pool()));
}

void Synapse::check_and_save_history(int step) {
  if ((step - last_flush_step_) >= history_interval_steps_) {
    std::ostringstream oss;
    oss << "./spiking_results/" << synapse_name_ << "_" << std::setw(4)
        << std::setfill('0') << flush_count_ << ".bin";
    save_history_to_bin(oss.str());
    clear_history();
    last_flush_step_ = step;
    flush_count_++;
  }
}

void Synapse::save_history_to_bin(const std::string& filename) {
  const int32_t header_len = static_cast<int32_t>(header_str_.size());
  const int32_t n = static_cast<int32_t>(pre_spike_history_.size());

  std::ofstream ofs(filename, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  ofs.write(header_str_.data(), header_len);
  ofs.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
  ofs.write(reinterpret_cast<const char*>(pre_spike_history_.data()),
            n * sizeof(uint8_t));
  ofs.write(reinterpret_cast<const char*>(post_spike_history_.data()),
            n * sizeof(uint8_t));
  ofs.write(reinterpret_cast<const char*>(weight_history_.data()),
            n * sizeof(float));
  ofs.write(reinterpret_cast<const char*>(receptor_pool_history_.data()),
            n * sizeof(float));
  ofs.close();
}

void Synapse::clear_history() {
  pre_spike_history_.clear();
  post_spike_history_.clear();
  weight_history_.clear();
  receptor_pool_history_.clear();
}

// ------------------------------ ExcitatorySynapse ------------------------------
ExcitatorySynapse::ExcitatorySynapse(std::shared_ptr<Neuron> pre,
                                     std::shared_ptr<Neuron> post, double w,
                                     int flush_interval_sec, double dt,
                                     double A_plus, double A_minus, double tau_plus,
                                     double tau_minus)
    : Synapse(std::move(pre), std::move(post), w, flush_interval_sec, dt, A_plus,
              A_minus, tau_plus, tau_minus) {
  const std::string& prestr = pre_neuron_->get_id_str();
  const std::string& poststr = post_neuron_->get_id_str();
  synapse_name_ = "syn_" + prestr.substr(0, prestr.size() - 4) + "_to_" +
                  poststr.substr(0, poststr.size() - 4) + "_AMPA";

  json header = {{"pre_layer_id", pre_neuron_->get_layer_id()},
                 {"pre_neuron_id", pre_neuron_->get_neuron_id()},
                 {"post_layer_id", post_neuron_->get_layer_id()},
                 {"post_neuron_id", post_neuron_->get_neuron_id()},
                 {"type", "AMPA"},
                 {"initial_weight", w},
                 {"A_plus", A_plus_},
                 {"A_minus", A_minus_},
                 {"tau_plus", tau_plus_},
                 {"tau_minus", tau_minus_},
                 {"dt", dt_}};
  header_str_ = header.dump();
}

void ExcitatorySynapse::transmit(int t) {
  if (pre_neuron_->is_spiking()) {
    post_neuron_->add_current(weight_);
  }
  record_current_state();
  check_and_save_history(t);
}

void ExcitatorySynapse::apply_stdp(int t) {
  if (post_neuron_->is_freeze()) return;

  // スパイク時刻更新（直近 kMaxHistory_ 件のみ）
  if (pre_neuron_->is_spiking()) {
    pre_spike_times_.push_back(t);
    if (static_cast<int>(pre_spike_times_.size()) > kMaxHistory_) {
      pre_spike_times_.erase(pre_spike_times_.begin());
    }
  }
  if (post_neuron_->is_spiking()) {
    post_spike_times_.push_back(t);
    if (static_cast<int>(post_spike_times_.size()) > kMaxHistory_) {
      post_spike_times_.erase(post_spike_times_.begin());
    }
  }

  // pre基準: post先行→LTD
  if (pre_neuron_->is_spiking()) {
    for (int t_post : post_spike_times_) {
      const double delta_t = (t_post - t) * dt_;  // Δt = t_post - t_pre
      if (delta_t == 0.0) continue;
      const double delta_w = -A_minus_ * std::exp(delta_t / tau_minus_);
      post_neuron_->restore_receptors(-delta_w);  // 減少分を返却＝備蓄回復
      weight_ += delta_w;
    }
  }

  // post基準: pre先行→LTP
  if (post_neuron_->is_spiking()) {
    for (int t_pre : pre_spike_times_) {
      const double delta_t = (t - t_pre) * dt_;
      const double delta_w = A_plus_ * std::exp(-delta_t / tau_plus_);
      const double actual_dw = post_neuron_->request_receptors(delta_w);  // 資源制約
      weight_ += actual_dw;
    }
  }

  // 次段 Inhibitory が参照する平均へ反映（pre側に集約）
  pre_neuron_->add_sum_weight(weight_);
}

// ------------------------------ InhibitorySynapse ------------------------------
InhibitorySynapse::InhibitorySynapse(std::shared_ptr<Neuron> pre,
                                     std::shared_ptr<Neuron> post, double w,
                                     int flush_interval_sec, double dt,
                                     double A_plus, double A_minus, double tau_plus,
                                     double tau_minus)
    : Synapse(std::move(pre), std::move(post), w, flush_interval_sec, dt, A_plus,
              A_minus, tau_plus, tau_minus) {
  const std::string& prestr = pre_neuron_->get_id_str();
  const std::string& poststr = post_neuron_->get_id_str();
  synapse_name_ = "syn_" + prestr.substr(0, prestr.size() - 4) + "_to_" +
                  poststr.substr(0, poststr.size() - 4) + "_GABA";

  json header = {{"pre_layer_id", pre_neuron_->get_layer_id()},
                 {"pre_neuron_id", pre_neuron_->get_neuron_id()},
                 {"post_layer_id", post_neuron_->get_layer_id()},
                 {"post_neuron_id", post_neuron_->get_neuron_id()},
                 {"type", "GABA"},
                 {"initial_weight", w},
                 {"A_plus", A_plus_},
                 {"A_minus", A_minus_},
                 {"tau_plus", tau_plus_},
                 {"tau_minus", tau_minus_},
                 {"dt", dt_}};
  header_str_ = header.dump();
}

void InhibitorySynapse::transmit(int t) {
  if (pre_neuron_->is_spiking()) {
    post_neuron_->add_current(-weight_);
  }
  record_current_state();
  check_and_save_history(t);
}
