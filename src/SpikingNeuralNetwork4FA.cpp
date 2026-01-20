#include "SpikingNeuralNetwork4FA.hpp"

SpikingNeuralNetwork4FA::SpikingNeuralNetwork4FA(uint32_t seed)
  : SpikingNeuralNetwork(seed) {}

double SpikingNeuralNetwork4FA::clip_(double x, double a) {
  if (a <= 0.0) return x;
  if (x > a) return a;
  if (x < -a) return -a;
  return x;
}

std::vector<double> SpikingNeuralNetwork4FA::normalize_prob_(std::vector<double> v) {
  double s = 0.0;
  for (double x : v) s += x;
  if (s <= 0.0) {
    v.assign(v.size(), 1.0 / std::max<size_t>(1, v.size()));
    return v;
  }
  for (double& x : v) x /= s;
  return v;
}

SpikingNeuralNetwork4FA::StdpFreezeState SpikingNeuralNetwork4FA::capture_freeze_state_() const {
  StdpFreezeState st;
  // layerにニューロンがいないケースは想定しないが、一応ガード
  if (kLayerHidden < static_cast<int>(neurons.size()) && !neurons[static_cast<size_t>(kLayerHidden)].empty()) {
    st.hidden_freeze = neurons[static_cast<size_t>(kLayerHidden)][0]->is_freeze();
  }
  if (kLayerOutput < static_cast<int>(neurons.size()) && !neurons[static_cast<size_t>(kLayerOutput)].empty()) {
    st.output_freeze = neurons[static_cast<size_t>(kLayerOutput)][0]->is_freeze();
  }
  return st;
}

void SpikingNeuralNetwork4FA::restore_freeze_state_(const StdpFreezeState& st) {
  // freeze==true => disableSTDP, false => enableSTDP
  if (st.hidden_freeze) disableSTDP(kLayerHidden); else enableSTDP(kLayerHidden);
  if (st.output_freeze) disableSTDP(kLayerOutput); else enableSTDP(kLayerOutput);
}

void SpikingNeuralNetwork4FA::build() {
  build(Config{});
}

void SpikingNeuralNetwork4FA::build(const Config& cfg) {
  cfg_ = cfg;

  neurons.clear();
  synapses.clear();
  current_step = 0;

  add_neurons<InputNeuron>(kLayerInput, kNInput,
                           cfg_.inp.max_receptor,
                           cfg_.inp.flush_interval_sec,
                           cfg_.inp.dt);

  add_neurons<SpikingNeuron>(kLayerHidden, kNHidden,
                             cfg_.hidden.max_receptor,
                             cfg_.hidden.flush_interval_sec,
                             cfg_.hidden.tau_m,
                             cfg_.hidden.v_rest,
                             cfg_.hidden.v_reset,
                             cfg_.hidden.v_th,
                             cfg_.hidden.refractory_period,
                             cfg_.hidden.dt);

  add_neurons<OutputNeuron>(kLayerOutput, kNOutput,
                            cfg_.out.max_receptor,
                            cfg_.out.flush_interval_sec,
                            cfg_.out.tau_m,
                            cfg_.out.v_rest,
                            cfg_.out.v_reset,
                            cfg_.out.v_th,
                            cfg_.out.refractory_period,
                            cfg_.out.dt);

  const int layers = static_cast<int>(neurons.size());
  synapses.assign(std::max(0, layers - 1), {});

  if (!cfg_.random_connect) {
    add_full_connect<ExcitatorySynapse>(
      kLayerInput, kLayerHidden,
      cfg_.in_to_hidden.weight,
      cfg_.in_to_hidden.flush_interval_sec,
      cfg_.in_to_hidden.dt,
      cfg_.in_to_hidden.A_plus,
      cfg_.in_to_hidden.A_minus,
      cfg_.in_to_hidden.tau_plus,
      cfg_.in_to_hidden.tau_minus
    );

    add_full_connect<ExcitatorySynapse>(
      kLayerHidden, kLayerOutput,
      cfg_.hidden_to_out.weight,
      cfg_.hidden_to_out.flush_interval_sec,
      cfg_.hidden_to_out.dt,
      cfg_.hidden_to_out.A_plus,
      cfg_.hidden_to_out.A_minus,
      cfg_.hidden_to_out.tau_plus,
      cfg_.hidden_to_out.tau_minus
    );
  } else {
    add_random_connect<ExcitatorySynapse>(
      kLayerInput, kLayerHidden,
      cfg_.n_per_post,
      cfg_.in_to_hidden.weight,
      cfg_.in_to_hidden.flush_interval_sec,
      cfg_.in_to_hidden.dt,
      cfg_.in_to_hidden.A_plus,
      cfg_.in_to_hidden.A_minus,
      cfg_.in_to_hidden.tau_plus,
      cfg_.in_to_hidden.tau_minus
    );

    add_random_connect<ExcitatorySynapse>(
      kLayerHidden, kLayerOutput,
      cfg_.n_per_post,
      cfg_.hidden_to_out.weight,
      cfg_.hidden_to_out.flush_interval_sec,
      cfg_.hidden_to_out.dt,
      cfg_.hidden_to_out.A_plus,
      cfg_.hidden_to_out.A_minus,
      cfg_.hidden_to_out.tau_plus,
      cfg_.hidden_to_out.tau_minus
    );
  }

  sort_synapses_by_type();

  last_input_rates_.assign(static_cast<size_t>(kNInput), cfg_.rate_off_hz);
  clear_episode_buffers();
  baseline_initialized_ = false;
  baseline_ = 0.0;
}

std::vector<double>
SpikingNeuralNetwork4FA::encode_obs01_to_rates_hz_(const std::vector<uint8_t>& obs01) const {
  if (static_cast<int>(obs01.size()) != kNInput) {
    throw std::runtime_error("set_observation_7x7_01: obs01.size must be 49");
  }
  std::vector<double> rates(static_cast<size_t>(kNInput), cfg_.rate_off_hz);
  for (int i = 0; i < kNInput; ++i) {
    rates[static_cast<size_t>(i)] =
      (obs01[static_cast<size_t>(i)] ? cfg_.rate_on_hz : cfg_.rate_off_hz);
  }
  return rates;
}

void SpikingNeuralNetwork4FA::set_input_rates_hz(const std::vector<double>& rates_hz) {
  if (static_cast<int>(rates_hz.size()) != kNInput) {
    throw std::runtime_error("set_input_rates_hz: rates_hz.size must be 49");
  }
  last_input_rates_ = rates_hz;
  set_inputs(rates_hz);
}

void SpikingNeuralNetwork4FA::set_observation_7x7_01(const std::vector<uint8_t>& obs01) {
  set_input_rates_hz(encode_obs01_to_rates_hz_(obs01));
}

std::vector<double> SpikingNeuralNetwork4FA::run_and_get_action_scores(int steps, bool as_rate_hz) {
  std::vector<int> counts(static_cast<size_t>(kNOutput), 0);

  for (int t = 0; t < steps; ++t) {
    step();
    for (int i = 0; i < kNOutput; ++i) {
      if (neurons[static_cast<size_t>(kLayerOutput)][static_cast<size_t>(i)]->is_spiking()) {
        counts[static_cast<size_t>(i)]++;
      }
    }
  }

  std::vector<double> scores(static_cast<size_t>(kNOutput), 0.0);
  if (!as_rate_hz) {
    for (int i = 0; i < kNOutput; ++i) {
      scores[static_cast<size_t>(i)] = static_cast<double>(counts[static_cast<size_t>(i)]);
    }
    return scores;
  }

  const double T = steps * cfg_.out.dt;
  if (T > 0.0) {
    for (int i = 0; i < kNOutput; ++i) {
      scores[static_cast<size_t>(i)] = static_cast<double>(counts[static_cast<size_t>(i)]) / T;
    }
  }
  return scores;
}

int SpikingNeuralNetwork4FA::act_greedy(int steps, bool as_rate_hz) {
  const auto scores = run_and_get_action_scores(steps, as_rate_hz);
  auto it = std::max_element(scores.begin(), scores.end());
  return static_cast<int>(std::distance(scores.begin(), it));
}

// ---------------- episode control ----------------
void SpikingNeuralNetwork4FA::clear_episode_buffers() {
  action_hist_.clear();
}

void SpikingNeuralNetwork4FA::begin_episode(bool freeze_plasticity) {
  clear_episode_buffers();
  if (freeze_plasticity) {
    // episode中は plasticity を止める（postとしてのSTDP停止）
    disableSTDP(kLayerHidden);
    disableSTDP(kLayerOutput);
  }
}

void SpikingNeuralNetwork4FA::record_action(int action_id) {
  if (0 <= action_id && action_id < kNOutput) action_hist_.push_back(action_id);
}

double SpikingNeuralNetwork4FA::end_episode_and_get_advantage(double episode_reward) {
  if (!cfg_.reward_gate.use_baseline) return episode_reward;

  if (!baseline_initialized_) {
    baseline_ = episode_reward;
    baseline_initialized_ = true;
    return 0.0;
  }

  const double adv = episode_reward - baseline_;
  const double b = cfg_.reward_gate.baseline_beta;
  baseline_ = (1.0 - b) * baseline_ + b * episode_reward;
  return adv;
}

// ---------------- target from actions ----------------
std::vector<double> SpikingNeuralNetwork4FA::make_target_from_actions_() const {
  // 行動履歴が無い場合は一様
  if (action_hist_.empty()) {
    return std::vector<double>(static_cast<size_t>(kNOutput), 1.0 / kNOutput);
  }

  std::vector<double> cnt(static_cast<size_t>(kNOutput), 1e-6); // avoid all-zero
  for (int a : action_hist_) {
    if (0 <= a && a < kNOutput) cnt[static_cast<size_t>(a)] += 1.0;
  }
  return normalize_prob_(cnt);
}

// ---------------- (2) teacher phase ----------------
void SpikingNeuralNetwork4FA::teacher_phase(const std::vector<double>& target_prob,
                                            int steps,
                                            double current_scale,
                                            bool freeze_hidden) {
  if (static_cast<int>(target_prob.size()) != kNOutput) {
    throw std::runtime_error("teacher_phase: target_prob.size must be 8");
  }

  const auto prev = capture_freeze_state_();

  // teacher中だけ plasticity ON（hidden->out を更新したい）
  enableSTDP(kLayerOutput);
  if (freeze_hidden) disableSTDP(kLayerHidden);

  // 直近観測を replay
  set_inputs(last_input_rates_);

  for (int t = 0; t < steps; ++t) {
    for (int j = 0; j < kNOutput; ++j) {
      const double inj = current_scale * target_prob[static_cast<size_t>(j)];
      neurons[static_cast<size_t>(kLayerOutput)][static_cast<size_t>(j)]->add_current(inj);
    }
    SpikingNeuralNetwork::step(); // built-in STDP
  }

  // 呼び出し側がタイミング制御できるよう、状態は元に戻す
  restore_freeze_state_(prev);
}

void SpikingNeuralNetwork4FA::teacher_phase_from_actions(double /*advantage*/,
                                                         std::optional<int> steps_override,
                                                         std::optional<double> current_scale_override) {
  if (!cfg_.teacher.enabled) return;
  const int steps = steps_override.has_value() ? steps_override.value() : cfg_.teacher.teacher_steps;
  const double scale = current_scale_override.has_value() ? current_scale_override.value() : cfg_.teacher.current_scale;

  const auto target = make_target_from_actions_();
  teacher_phase(target, steps, scale, /*freeze_hidden=*/true);
}

// ---------------- (1) reward-gated learning phase (LTP/LTD bias by timing) ----------------
void SpikingNeuralNetwork4FA::reward_gated_learning_phase_from_actions(
    double advantage, std::optional<int> steps_override) {

  if (!cfg_.reward_gate.enabled) return;

  const double adv = clip_(advantage, cfg_.reward_gate.adv_clip);
  const int total = steps_override.has_value() ? steps_override.value()
                                               : cfg_.reward_gate.learning_steps;
  const int stepsA = static_cast<int>(std::round(cfg_.reward_gate.phaseA_ratio * total));
  const int stepsB = std::max(0, total - stepsA);

  const auto prev = capture_freeze_state_();

  // plasticity ON（hidden->out を更新したい）
  enableSTDP(kLayerOutput);
  disableSTDP(kLayerHidden);

  // ターゲット分布は「選択した行動分布」を使う（負advでも反教師にはしない）
  const auto target = make_target_from_actions_();

  // 注入強度は |adv| でスケール（符号は時間順序で表現）
  const double scale = cfg_.reward_gate.current_scale * std::abs(adv);

  if (adv >= 0.0) {
    // -------- LTP寄せ：pre -> post --------
    // Phase A: input only (pre earlier)
    set_inputs(last_input_rates_);
    for (int t = 0; t < stepsA; ++t) {
      SpikingNeuralNetwork::step();
    }

    // Phase B: +output injection (post later)
    set_inputs(last_input_rates_);
    for (int t = 0; t < stepsB; ++t) {
      for (int j = 0; j < kNOutput; ++j) {
        neurons[static_cast<size_t>(kLayerOutput)][static_cast<size_t>(j)]
          ->add_current(scale * target[static_cast<size_t>(j)]);
      }
      SpikingNeuralNetwork::step();
    }

  } else {
    // -------- LTD寄せ：post -> pre --------
    // Phase A: output injection with suppressed input (post earlier)
    std::vector<double> suppressed = last_input_rates_;
    for (double& v : suppressed) v *= cfg_.reward_gate.pre_suppress; // 0.0 なら input をほぼ止める
    set_inputs(suppressed);

    for (int t = 0; t < stepsA; ++t) {
      for (int j = 0; j < kNOutput; ++j) {
        neurons[static_cast<size_t>(kLayerOutput)][static_cast<size_t>(j)]
          ->add_current(scale * target[static_cast<size_t>(j)]);
      }
      SpikingNeuralNetwork::step();
    }

    // Phase B: input only (pre later) with no injection
    set_inputs(last_input_rates_);
    for (int t = 0; t < stepsB; ++t) {
      SpikingNeuralNetwork::step();
    }
  }

  // 状態を元に戻す
  restore_freeze_state_(prev);
}

// ---------------- (3) sleep ----------------
void SpikingNeuralNetwork4FA::sleep_run(int steps, double rate_hz, bool enable_stdp) {
  const auto prev = capture_freeze_state_();

  if (enable_stdp) {
    // sleep中に LTD を期待するので plasticity ON（少なくとも output はON）
    enableSTDP(kLayerHidden);
    enableSTDP(kLayerOutput);
  }

  // inputはベースライン（0にはしない）
  std::vector<double> base_in(static_cast<size_t>(kNInput), cfg_.rate_off_hz);
  set_inputs(base_in);

  set_sleep_mode_all_spiking(true, rate_hz);
  run(steps);
  set_sleep_mode_all_spiking(false, 0.0);

  restore_freeze_state_(prev);
}

void SpikingNeuralNetwork4FA::sleep_phase(const SleepConfig& scfg) {
  if (!scfg.enabled) return;
  sleep_run(scfg.sleep_steps, scfg.sleep_rate_hz, scfg.enable_stdp_during_sleep);
}
