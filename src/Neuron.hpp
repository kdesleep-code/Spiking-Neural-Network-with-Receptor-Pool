#pragma once
/**
 * @file Neuron.hpp
 * @brief ニューロン・シナプス（興奮性／抑制性）クラス定義
 *
 * 設計ポイント
 * - STDPは Excitatory → Inhibitory の順で適用
 * - 睡眠モード（Random Firing）:
 *    - Spiking/Output が sleep のとき、入力電流は力学的には 0（無視）
 *    - p = rate*dt で強制スパイク
 *    - post 側 STDP を freeze（OFFで解除）
 *    - ★ログ専用の入力 input_current_log_ にのみパルスを書き込む
 */

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <random>
#include <optional>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <memory>

#include "./static_lib/json.hpp"
using json = nlohmann::json;

// ========== Neuron (抽象基底) ==========
class Neuron {
protected:
  int layer_id_;
  int neuron_id_;
  int flush_count_;
  std::string id_str_;

  // 力学用の入力（膜電位更新に効く）
  double input_current_;
  // ★ログ専用入力（膜電位更新には一切使わない）
  double input_current_log_;

  double membrane_potential_;
  double receptor_pool_;
  double max_receptor_;

  std::vector<uint8_t> spike_history_;
  std::vector<double> membrane_potential_history_;
  std::vector<double> receptor_pool_history_;
  std::vector<double> input_current_history_;

  bool record_history_;
  int history_interval_steps_;
  int last_flush_steps_;
  int flush_interval_sec_;
  bool force_spike_flag_;

  // 次段 Inhibitory が参照する平均のための集約
  double sum_weight_;
  int connection_count_;

  std::string header_str_;
  bool freeze_;  // post としての STDP 停止

public:
  Neuron(int layer_id, int neuron_id, double max_receptor,
         int flush_interval_sec, double dt);
  virtual ~Neuron();

  virtual void step(int t) = 0;

  virtual void reset();

  // 受容体資源（postの備蓄）
  void restore_receptors(double amount);
  double request_receptors(double amount);
  double get_receptor_pool() const;

  // ログ制御
  void clear_history();
  void enable_history_recording(bool flag);
  void record_current_state();
  void check_and_save_history(int step);
  void save_history_to_bin(const std::string& filename);

  // 入出力・平均重み
  inline void add_current(double v) {            // 力学+ログの両方に反映（通常時）
    input_current_ += v;
    input_current_log_ += v;
  }
  inline void clear_current() {                  // 両方クリア
    input_current_ = 0.0;
    input_current_log_ = 0.0;
  }
  // ★ログだけに加算（膜電位には影響させない）
  inline void add_current_log_only(double v) {
    input_current_log_ += v;
  }

  inline void add_sum_weight(double w) { sum_weight_ += w; connection_count_++; }
  inline void clear_sum_weight() { sum_weight_ = 0.0; connection_count_ = 0; }

  // 基本情報
  inline int get_layer_id() const { return layer_id_; }
  inline int get_neuron_id() const { return neuron_id_; }
  inline const std::string& get_id_str() const { return id_str_; }

  // STDP freeze（post側としての更新無効化）
  inline void set_freeze(bool f) { freeze_ = f; }
  inline bool is_freeze() const { return freeze_; }

  // 型コード（グラフ出力用）: 0=Input, 1=Spiking, 2=Output
  virtual uint8_t type_code() const = 0;

  // 発火フラグ（派生で実装）
  virtual bool is_spiking() const = 0;
};

// ========== InputNeuron ==========
class InputNeuron : public Neuron {
private:
  double input_value_;
  double input_rate_;
  double dt_;
  bool spiking_now_;

  uint32_t seed_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform01_;

public:
  InputNeuron(int layer_id, int neuron_id, double max_receptor,
              int flush_interval_sec, double dt, std::optional<uint32_t> seed_opt = std::nullopt);

  void set_input(double value); // value[Hz] をそのまま発火率に

  void step(int t) override;
  void reset() override;

  bool is_spiking() const override;
  uint8_t type_code() const override { return 0; }
};

// ========== SpikingNeuron（LIF） ==========
class SpikingNeuron : public Neuron {
protected:
  // LIF パラメータ
  double tau_m_;
  double v_rest_;
  double v_reset_;
  double v_th_;
  double refractory_period_;

  bool spiking_now_;
  double refractory_time_left_;
  double dt_;

  // 睡眠（Random Firing）関連
  bool sleep_mode_ = false;
  double sleep_rate_hz_ = 0.0;
  uint32_t sleep_seed_ = 0;
  std::mt19937 sleep_rng_;
  std::uniform_real_distribution<double> sleep_u01_{0.0, 1.0};

public:
  SpikingNeuron(int layer_id, int neuron_id, double max_receptor,
                int flush_interval_sec, double tau_m, double v_rest,
                double v_reset, double v_th, double refractory_period, double dt);

  void step(int t) override;
  void reset() override;

  bool is_spiking() const override;
  uint8_t type_code() const override { return 1; }

  // Sleep モード切替（on=trueで post STDP freeze／offで解除）
  void set_sleep_mode(bool on, double rate_hz = 0.0);
  bool is_sleep_mode() const { return sleep_mode_; }
  double sleep_rate_hz() const { return sleep_rate_hz_; }
};

// ========== OutputNeuron ==========
class OutputNeuron : public SpikingNeuron {
public:
  OutputNeuron(int layer_id, int neuron_id, double max_receptor,
               int flush_interval_sec, double tau_m, double v_rest,
               double v_reset, double v_th, double refractory_period, double dt);

  uint8_t type_code() const override { return 2; }
};

// ========== Synapse（抽象基底） ==========
class Synapse {
protected:
  std::shared_ptr<Neuron> pre_neuron_;
  std::shared_ptr<Neuron> post_neuron_;
  double weight_;

  std::string synapse_name_;

  std::vector<uint8_t> pre_spike_history_;
  std::vector<uint8_t> post_spike_history_;
  std::vector<float> weight_history_;
  std::vector<float> receptor_pool_history_;

  int history_interval_steps_;
  int last_flush_step_;
  int flush_count_;

  // STDP パラメータ
  double A_plus_, A_minus_, tau_plus_, tau_minus_;
  double dt_;

  // スパイク時刻（最近 kMaxHistory_ 件）
  static constexpr int kMaxHistory_ = 64;
  std::vector<int> pre_spike_times_;
  std::vector<int> post_spike_times_;

  std::string header_str_;

public:
  Synapse(std::shared_ptr<Neuron> pre, std::shared_ptr<Neuron> post, double w,
          int flush_interval_sec, double dt,
          double A_plus, double A_minus, double tau_plus, double tau_minus);

  virtual ~Synapse() = default;

  virtual void transmit(int t) = 0;
  virtual void apply_stdp(int t) = 0;

  void record_current_state();
  void check_and_save_history(int step);
  void save_history_to_bin(const std::string& filename);
  void clear_history();

  inline std::shared_ptr<Neuron> get_pre_neuron() const { return pre_neuron_; }
  inline std::shared_ptr<Neuron> get_post_neuron() const { return post_neuron_; }
  inline double get_weight() const { return weight_; }

  // 型コード（グラフ出力用）: 0=AMPA, 1=GABA
  virtual uint8_t type_code() const = 0;
};

// ========== ExcitatorySynapse（AMPA） ==========
class ExcitatorySynapse : public Synapse {
public:
  ExcitatorySynapse(std::shared_ptr<Neuron> pre, std::shared_ptr<Neuron> post, double w,
                    int flush_interval_sec, double dt,
                    double A_plus, double A_minus, double tau_plus, double tau_minus);

  void transmit(int t) override;
  void apply_stdp(int t) override;
  uint8_t type_code() const override { return 0; }
};

// ========== AMPA for Output Neuron ===========

class RExcitatorySynapse : public ExcitatorySynapse {
public:
  using ExcitatorySynapse::ExcitatorySynapse; // 既存コンストラクタ継承

  // δ（TD誤差などの報酬信号）を外部から注入
  inline void set_delta(double d) { delta_ = d; }
  inline void set_rstdp_params(double eta, double tau_pre, double tau_post, double tau_elig) {
    eta_ = eta; tau_pre_ = tau_pre; tau_post_ = tau_post; tau_elig_ = tau_elig;
  }

  // ★ 純STDPではなく「R-STDP」をここで実行（Output用は常にこれ）
  void apply_stdp(int /*t*/) override;

  // 可視化/保存互換のため AMPA と同じコードを返す
  uint8_t type_code() const override { return 0; }

private:
  // eligibility-trace と痕跡
  double pre_trace_  = 0.0;
  double post_trace_ = 0.0;
  double elig_       = 0.0;

  // R-STDPハイパラ
  double eta_ = 1e-3;
  double tau_pre_  = 0.02;  // [s]
  double tau_post_ = 0.02;  // [s]
  double tau_elig_ = 0.10;  // [s]

  // 報酬信号（setterで更新）
  double delta_ = 0.0;
};

// ========== InhibitorySynapse（GABA） ==========
class InhibitorySynapse : public Synapse {
public:
  InhibitorySynapse(std::shared_ptr<Neuron> pre, std::shared_ptr<Neuron> post, double w,
                    int flush_interval_sec, double dt,
                    double A_plus, double A_minus, double tau_plus, double tau_minus);

  void transmit(int t) override;

  void apply_stdp(int t) override {
    (void)t;  // デフォルトは何もしない（必要に応じて拡張）
  }

  uint8_t type_code() const override { return 1; }
};
