#pragma once
/**
 * @file SpikingNeuralNetwork4FA.hpp
 * @brief 49-100-8 固定構造の function approximation / RL 用ラッパ
 *
 * - Input : 49  (7x7, 0/1 observation -> Poisson rate[Hz])
 * - Hidden: 100 (SpikingNeuron)
 * - Output: 8   (OutputNeuron) : 8-direction action scores (spike count or Hz)
 *
 * 学習系（外部からタイミング制御）:
 *  (1) Reward-gated STDP learning phase:
 *      episode末に advantage を渡して呼ぶ。weight直書きなしで、既存STDPに任せる。
 *      adv>0: pre->post を寄せて LTP 側
 *      adv<0: post->pre を寄せて LTD 側（プール削りの意図）
 *
 *  (2) Teacher phase:
 *      target_prob を指定して output に電流注入しながら STDP を回す。
 *
 *  (3) Sleep:
 *      ランダム発火で回すだけ（pruneなし）。sleep中はSTDPをONにして LTD を期待。
 */

#include <cstdint>
#include <vector>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "SpikingNeuralNetwork.hpp"

class SpikingNeuralNetwork4FA : public SpikingNeuralNetwork {
public:
  // ----- layer ids -----
  static constexpr int kLayerInput  = 0;
  static constexpr int kLayerHidden = 1;
  static constexpr int kLayerOutput = 2;

  // ----- neuron counts (固定) -----
  static constexpr int kNInput  = 49;   // 7x7
  static constexpr int kNHidden = 100;
  static constexpr int kNOutput = 8;    // 8 directions

  // 8方向の並び（必要なら外側の解釈と合わせて変更）
  enum Action8 : int {
    N  = 0, NE = 1, E  = 2, SE = 3,
    S  = 4, SW = 5, W  = 6, NW = 7
  };

  // ----- parameter packs -----
  struct NeuronParamsInput {
    double max_receptor = 10.0;
    int flush_interval_sec = 100;
    double dt = 0.001;
  };

  struct NeuronParamsSpiking {
    double max_receptor = 10.0;
    int flush_interval_sec = 100;
    double tau_m = 20.0;
    double v_rest = 0.0;
    double v_reset = 0.0;
    double v_th = 1.0;
    double refractory_period = 0.002;
    double dt = 0.001;
  };

  struct SynapseParams {
    double weight = 0.05;
    int flush_interval_sec = 100;
    double dt = 0.001;
    double A_plus = 0.01;
    double A_minus = 0.012;
    double tau_plus = 20.0;
    double tau_minus = 20.0;
  };

  struct RewardGateConfig {
    bool enabled = true;

    // baseline: advantage = reward - baseline
    bool use_baseline = true;
    double baseline_beta = 0.01;

    // learning phase length
    int learning_steps = 800;

    // two-phase split ratio (0..1): PhaseA = ratio*learning_steps, PhaseB = rest
    // adv>0:
    //   PhaseA: input only (pre earlier)
    //   PhaseB: +output injection (post later) -> LTP bias
    // adv<0:
    //   PhaseA: output injection with suppressed input (post earlier)
    //   PhaseB: input only (pre later) -> LTD bias
    double phaseA_ratio = 0.5;

    // output injection scale; actual = current_scale * |adv|
    double current_scale = 1.0;

    // clip advantage magnitude to avoid too strong injection
    double adv_clip = 5.0;

    // for adv<0 PhaseA: suppress input by this factor (0.0: off, 1.0: normal)
    double pre_suppress = 0.0;
  };

  struct TeacherConfig {
    bool enabled = true;
    int teacher_steps = 500;
    double current_scale = 1.0; // injection scale
  };

  struct SleepConfig {
    bool enabled = true;
    int sleep_steps = 2000;
    double sleep_rate_hz = 5.0; // random firing
    bool enable_stdp_during_sleep = true; // LTDを期待するなら true
  };

  struct Config {
    NeuronParamsInput inp;
    NeuronParamsSpiking hidden;
    NeuronParamsSpiking out;

    SynapseParams in_to_hidden;
    SynapseParams hidden_to_out;

    bool random_connect = false;
    int n_per_post = 10;

    // observation(0/1) -> input rates[Hz]
    double rate_on_hz  = 80.0;
    double rate_off_hz = 2.0;

    // 1 env-step あたりのシミュレーション長（呼び出し側の便宜用）
    int default_steps_per_env_step = 2000;

    RewardGateConfig reward_gate;
    TeacherConfig teacher;
    SleepConfig sleep;
  };

public:
  explicit SpikingNeuralNetwork4FA(uint32_t seed = 0);

  // --- build fixed 49-100-8 ---
  // NOTE: GCC互換のため default 引数は使わない（Config{} をデフォルト引数にするとエラーになるケースがある）
  void build();                 // default config
  void build(const Config& cfg); // explicit config

  // --- public read-only helpers (for tests etc.) ---
  int num_layers() const { return static_cast<int>(neurons.size()); }
  int layer_size(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(neurons.size())) return 0;
    return static_cast<int>(neurons[static_cast<size_t>(layer)].size());
  }

  // --- input ---
  void set_observation_7x7_01(const std::vector<uint8_t>& obs01);
  void set_input_rates_hz(const std::vector<double>& rates_hz);

  // --- output scores ---
  // as_rate_hz=true: Hz, false: spike count
  std::vector<double> run_and_get_action_scores(int steps, bool as_rate_hz = true);
  int act_greedy(int steps, bool as_rate_hz = true);

  // --- episode control (外側から呼ぶ) ---
  void begin_episode(bool freeze_plasticity = true);
  void record_action(int action_id);
  double end_episode_and_get_advantage(double episode_reward);
  void clear_episode_buffers();

  // --- (1) reward-gated learning phase (外側から呼ぶ) ---
  void reward_gated_learning_phase_from_actions(double advantage,
                                                std::optional<int> steps_override = std::nullopt);

  // --- (2) teacher phase (外側から呼ぶ) ---
  void teacher_phase(const std::vector<double>& target_prob,
                     int steps,
                     double current_scale,
                     bool freeze_hidden = true);

  void teacher_phase_from_actions(double advantage,
                                  std::optional<int> steps_override = std::nullopt,
                                  std::optional<double> current_scale_override = std::nullopt);

  // --- (3) sleep phase (外側から呼ぶ) ---
  void sleep_run(int steps, double rate_hz, bool enable_stdp);
  void sleep_phase(const SleepConfig& scfg);

  // debug
  double get_baseline() const { return baseline_; }
  const Config& get_config() const { return cfg_; }

private:
  Config cfg_{};

  // baseline
  double baseline_ = 0.0;
  bool baseline_initialized_ = false;

  // episode action history
  std::vector<int> action_hist_;

  // last input (for replay during learning/teacher/sleep)
  std::vector<double> last_input_rates_;

private:
  std::vector<double> encode_obs01_to_rates_hz_(const std::vector<uint8_t>& obs01) const;

  // action_hist_ から target 分布(size=8)を作る（advの符号で分布を変えない：負advは時間順序でLTD寄せ）
  std::vector<double> make_target_from_actions_() const;

  static std::vector<double> normalize_prob_(std::vector<double> v);
  static double clip_(double x, double a);

  // utility: set STDP on/off for relevant layers, and restore
  struct StdpFreezeState {
    bool hidden_freeze = false;
    bool output_freeze = false;
  };
  StdpFreezeState capture_freeze_state_() const;
  void restore_freeze_state_(const StdpFreezeState& st);
};
