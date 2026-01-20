// tests/test_spikingnn4fa.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>

#include "SpikingNeuralNetwork4FA.hpp"

// ---------------- tiny test helpers ----------------
static void assert_true(bool cond, const char* msg) {
  if (!cond) throw std::runtime_error(msg);
}

static double mean_weight_hidden_to_out(SpikingNeuralNetwork4FA& snn) {
  // SpikingNeuralNetwork の protected member `synapses` にアクセスできない前提のため、
  // ラッパ側で weight を直接読むAPIが無い限り「平均重み」を正確に取れません。
  //
  // そこで、この簡易テストでは "出力スコアの平均" を proxy として用います。
  // ただし出力はノイズが大きいので、複数回平均します。
  //
  // ※もし `synapses` が public/protected でアクセスできるなら、
  //    ここで hidden->out のシナプス重み平均を取る実装に差し替えてください。
  return 0.0;
}

static std::vector<uint8_t> make_obs_all0() {
  return std::vector<uint8_t>(SpikingNeuralNetwork4FA::kNInput, 0);
}

static std::vector<uint8_t> make_obs_onehot(int idx) {
  std::vector<uint8_t> o(SpikingNeuralNetwork4FA::kNInput, 0);
  if (0 <= idx && idx < SpikingNeuralNetwork4FA::kNInput) o[(size_t)idx] = 1;
  return o;
}

static double avg_score_over_runs(SpikingNeuralNetwork4FA& snn,
                                  const std::vector<uint8_t>& obs,
                                  int runs,
                                  int steps_per_run,
                                  bool as_rate_hz = true) {
  double acc = 0.0;
  for (int r = 0; r < runs; ++r) {
    snn.set_observation_7x7_01(obs);
    auto scores = snn.run_and_get_action_scores(steps_per_run, as_rate_hz);
    double m = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    acc += m;
  }
  return acc / runs;
}

int main() {
  try {
    // ---------------- config ----------------
    SpikingNeuralNetwork4FA snn(123);

    SpikingNeuralNetwork4FA::Config cfg;
    cfg.default_steps_per_env_step = 400;  // テストなので短め

    // 入力符号化：0/1
    cfg.rate_on_hz  = 80.0;
    cfg.rate_off_hz = 2.0;

    // Reward-gated 学習（テストで反応が見えるように少し強め）
    cfg.reward_gate.enabled = true;
    cfg.reward_gate.use_baseline = false;     // テストでは baseline なしで直に adv を使う
    cfg.reward_gate.learning_steps = 600;
    cfg.reward_gate.phaseA_ratio = 0.5;
    cfg.reward_gate.current_scale = 2.0;      // 強め
    cfg.reward_gate.adv_clip = 3.0;
    cfg.reward_gate.pre_suppress = 0.0;       // adv<0 のとき input を止めて post->pre を作りやすく

    // teacher は任意（ここでは使わない）
    cfg.teacher.enabled = false;

    // sleep（pruneなし）
    cfg.sleep.enabled = true;
    cfg.sleep.sleep_steps = 400;
    cfg.sleep.sleep_rate_hz = 5.0;
    cfg.sleep.enable_stdp_during_sleep = true;

    // シナプス初期値：小さめ
    cfg.in_to_hidden.weight = 0.02;
    cfg.hidden_to_out.weight = 0.02;

    snn.build(cfg);

    // ---------------- shape check ----------------
    assert_true((int)snn.neurons.size() == 3, "Expected 3 layers");
    assert_true((int)snn.neurons[SpikingNeuralNetwork4FA::kLayerInput].size()  == SpikingNeuralNetwork4FA::kNInput,  "Input layer size mismatch");
    assert_true((int)snn.neurons[SpikingNeuralNetwork4FA::kLayerHidden].size() == SpikingNeuralNetwork4FA::kNHidden, "Hidden layer size mismatch");
    assert_true((int)snn.neurons[SpikingNeuralNetwork4FA::kLayerOutput].size() == SpikingNeuralNetwork4FA::kNOutput, "Output layer size mismatch");

    // ---------------- forward sanity ----------------
    {
      auto obs = make_obs_onehot(0);
      snn.set_observation_7x7_01(obs);
      auto scores = snn.run_and_get_action_scores(300, true);
      assert_true((int)scores.size() == SpikingNeuralNetwork4FA::kNOutput, "Score size mismatch");
      for (double v : scores) {
        assert_true(std::isfinite(v), "Score must be finite");
        assert_true(v >= 0.0, "Score must be non-negative");
      }
      std::cout << "[OK] forward sanity\n";
    }

    // ---------------- reward-gated LTP/LTD tendency (proxy by output mean) ----------------
    //
    // proxy: 同一入力に対する output 平均レートが
    //   adv>0学習後に上がりやすい / adv<0学習後に下がりやすい、を“傾向”として確認する。
    //
    // 注意：SNNノイズ大のため runs を複数回平均し、閾値は緩い。
    //
    const auto obs_train = make_obs_onehot(10);

    // baseline measurement
    double base = avg_score_over_runs(snn, obs_train, /*runs=*/6, /*steps=*/300, /*as_rate_hz=*/true);
    std::cout << "Base avg score: " << base << "\n";

    // positive advantage learning (LTP bias)
    {
      snn.begin_episode(true);
      // “行動履歴”を作る（たとえば E を選んだことにする）
      for (int k = 0; k < 20; ++k) snn.record_action(SpikingNeuralNetwork4FA::E);

      // 直近入力を設定（learning phase は last_input_rates_ を replay する）
      snn.set_observation_7x7_01(obs_train);

      // apply reward-gated learning with adv>0
      snn.reward_gated_learning_phase_from_actions(+2.0);

      double after_pos = avg_score_over_runs(snn, obs_train, 6, 300, true);
      std::cout << "After adv>0 avg score: " << after_pos << "\n";

      // 緩い期待：少なくとも極端に下がっていない
      assert_true(after_pos >= base * 0.85, "adv>0 learning made scores too small (unexpected)");
      std::cout << "[OK] reward-gated adv>0 executed\n";

      base = after_pos; // 次の比較用に更新（単純化）
    }

    // negative advantage learning (LTD bias)
    {
      snn.begin_episode(true);
      for (int k = 0; k < 20; ++k) snn.record_action(SpikingNeuralNetwork4FA::E);
      snn.set_observation_7x7_01(obs_train);

      snn.reward_gated_learning_phase_from_actions(-2.0);

      double after_neg = avg_score_over_runs(snn, obs_train, 6, 300, true);
      std::cout << "After adv<0 avg score: " << after_neg << "\n";

      // 緩い期待：少なくとも極端に上がっていない（LTD寄せ）
      assert_true(after_neg <= base * 1.25, "adv<0 learning increased scores too much (unexpected)");
      std::cout << "[OK] reward-gated adv<0 executed\n";

      base = after_neg;
    }

    // ---------------- sleep sanity ----------------
    {
      // sleep は実行できることだけ確認（prune無し）
      snn.sleep_phase(cfg.sleep);

      auto obs = make_obs_all0();
      double after_sleep = avg_score_over_runs(snn, obs, 4, 300, true);
      assert_true(std::isfinite(after_sleep), "Sleep produced non-finite output");
      std::cout << "[OK] sleep executed. avg score(all0): " << after_sleep << "\n";
    }

    std::cout << "All tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "TEST FAILED: " << e.what() << "\n";
    return 1;
  }
}
