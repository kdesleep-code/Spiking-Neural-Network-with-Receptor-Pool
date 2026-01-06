# SpikingNeuralNetwork4FA Usage Guide (49-100-8 Fixed Topology)

This document explains how to use `SpikingNeuralNetwork4FA`, a thin wrapper around the existing spiking neural network implementation, specialized for function approximation / reinforcement learning experiments with:

- **Input layer**: 49 neurons (flattened 7×7 binary observation)
- **Hidden layer**: 100 neurons
- **Output layer**: 8 neurons (continuous action scores via spike counts/rates)

The wrapper is designed so that **the caller controls the timing** of learning, teacher phases, and sleep phases via public methods.

---

## 1. Network Topology

`SpikingNeuralNetwork4FA` always constructs a 3-layer feedforward network:

- Layer 0: `InputNeuron` × 49
- Layer 1: `SpikingNeuron` × 100
- Layer 2: `OutputNeuron` × 8

The topology is fixed to match the environment interface:
- **49-dimensional observation** (7×7 local field of view)
- **8-direction actions** (N, NE, E, SE, S, SW, W, NW)

---

## 2. Observation Encoding (7×7 binary → Poisson rates)

Your environment provides a 49-length vector `obs01` with values in `{0,1}`.

The wrapper converts it into input firing rates (Hz) using:

- `1 -> rate_on_hz`
- `0 -> rate_off_hz`

Both are configurable via `Config`:

```cpp
SpikingNeuralNetwork4FA::Config cfg;
cfg.rate_on_hz  = 80.0;
cfg.rate_off_hz = 2.0;  // avoid pure silence
```

Then call:

```cpp
snn.set_observation_7x7_01(obs01);
```

---

## 3. Getting Action Scores (8 outputs)

The output layer has 8 neurons. Their spike activity is interpreted as continuous action scores.

You can obtain scores as:
- **Spike counts** over a window
- **Rates (Hz)** over a window (recommended for window-size invariance)

```cpp
auto scores_hz = snn.run_and_get_action_scores(steps, /*as_rate_hz=*/true);
int action_id  = snn.act_greedy(steps, /*as_rate_hz=*/true);
```

`act_greedy()` returns the argmax index of the 8 scores.

---

## 4. Episode Control (Caller-Driven)

This project assumes the reward is finalized **per episode**.

A typical loop is:

```cpp
snn.begin_episode(true);  // freeze plasticity during the episode

for (int t = 0; t < T; ++t) {
  snn.set_observation_7x7_01(obs01);

  // run the network for one environment step
  snn.run(cfg.default_steps_per_env_step);

  // select action
  int a = snn.act_greedy(200, true);
  snn.record_action(a);

  // env.step(a) ...
}

double adv = snn.end_episode_and_get_advantage(reward);
```

Notes:
- `begin_episode(true)` disables STDP in hidden/output as “post” neurons so that plasticity does not occur during the episode.
- `record_action(a)` stores the action sequence so that learning/teacher phases can derive a target distribution.

---

## 5. (1) Reward-Gated STDP (Episode-End Learning Phase)

The wrapper implements reward modulation **without directly writing synapse weights**.
Instead, it performs a short “learning phase” where:

- STDP is enabled (mainly for hidden → output).
- The output layer receives external current injections derived from the episode’s action distribution.
- The temporal order of activity biases STDP toward LTP or LTD.

### Advantage > 0: LTP-biased (pre → post)

1. Phase A: run with input only (pre activity happens first)
2. Phase B: run with output injection (post activity follows)

### Advantage < 0: LTD-biased (post → pre)

1. Phase A: inject output current while suppressing input (post first)
2. Phase B: run with input only (pre later)

This is implemented in:

```cpp
snn.reward_gated_learning_phase_from_actions(adv);
```

Key parameters:

```cpp
cfg.reward_gate.learning_steps = 800;
cfg.reward_gate.phaseA_ratio   = 0.5;
cfg.reward_gate.current_scale  = 1.0; // multiplied by |adv|
cfg.reward_gate.pre_suppress   = 0.0; // for negative adv Phase A
cfg.reward_gate.adv_clip       = 5.0;
```

If you want stronger LTD for negative advantage:
- Increase `phaseA_ratio` (e.g., 0.7)
- Set `pre_suppress` close to 0.0
- Increase `current_scale` carefully (too large may destabilize firing)

---

## 6. (2) Teacher Phase (Optional)

If you want to explicitly “teach” the network at the end of an episode, you can use:

- Teacher from recorded actions:
```cpp
snn.teacher_phase_from_actions(adv);
```

- Teacher with a custom target probability distribution:
```cpp
std::vector<double> target_prob(8, 0.0);
target_prob[2] = 1.0; // e.g., strongly encourage action E
snn.teacher_phase(target_prob, steps, current_scale);
```

Teacher phase runs STDP while injecting output currents proportional to `target_prob`.

---

## 7. (3) Sleep Phase (Random Firing Only)

The sleep phase is intentionally minimal here:
- It only enables random spiking activity and runs the network.
- No explicit pruning is performed.
- STDP can be enabled during sleep to allow LTD to occur naturally.

Usage:

```cpp
snn.sleep_phase(cfg.sleep);
```

Parameters:

```cpp
cfg.sleep.sleep_steps = 2000;
cfg.sleep.sleep_rate_hz = 5.0;
cfg.sleep.enable_stdp_during_sleep = true;
```

---

## 8. Practical Tips

- Avoid `rate_off_hz = 0.0` unless you know what you are doing. A small baseline keeps the network from becoming silent.
- Prefer output rates (Hz) rather than counts if you may change the scoring window length.
- Because spiking is stochastic, evaluation should average over multiple runs or longer windows.

---

## 9. Minimal Working Example

```cpp
SpikingNeuralNetwork4FA snn(123);
SpikingNeuralNetwork4FA::Config cfg;
snn.build(cfg);

for (int ep = 0; ep < 100; ++ep) {
  snn.begin_episode(true);

  for (int t = 0; t < 50; ++t) {
    std::vector<uint8_t> obs01(49, 0);
    obs01[0] = 1;

    snn.set_observation_7x7_01(obs01);
    snn.run(cfg.default_steps_per_env_step);

    int a = snn.act_greedy(200, true);
    snn.record_action(a);
  }

  double reward = /* episode reward */;
  double adv = snn.end_episode_and_get_advantage(reward);

  snn.reward_gated_learning_phase_from_actions(adv);
  snn.sleep_phase(cfg.sleep);
}
```

---

## 10. Summary

- `SpikingNeuralNetwork4FA` is a fixed 49-100-8 wrapper for RL-style experiments.
- The caller controls episode timing and explicitly triggers:
  - reward-gated learning
  - optional teacher phase
  - sleep phase
- Negative advantage is handled by a timing strategy that biases STDP toward LTD, supporting receptor-pool experiments.
