
#ifndef NEURON_HPP_
#define NEURON_HPP_

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

// ---------------------------------------------
// Minimal neuron/synapse interfaces
// ---------------------------------------------

class Neuron {
public:
  explicit Neuron(double dt) : dt_(dt) {}
  virtual ~Neuron() = default;

  // one simulation tick
  virtual void step(int /*t*/) {
    // Simple leaky-integrate placeholder
    v_ += input_current_;
    input_current_ = 0.0;
    // spike if over threshold
    spiking_ = (v_ >= v_th_);
    if (spiking_) v_ = v_reset_;
  }

  // receive synaptic current
  inline void add_input(double i) { input_current_ += i; }

  // spike state
  inline bool is_spiking() const { return spiking_; }

  // receptor pool interface (resource-constrained plasticity)
  // request receptors for LTP; returns actually granted amount (may be clipped)
  virtual double request_receptors(double want) {
    if (want <= 0.0) return 0.0;
    double got = std::min(want, receptor_pool_);
    receptor_pool_ -= got;
    return got;
  }
  // return receptors for LTD
  virtual void restore_receptors(double give) {
    receptor_pool_ += give;
    if (receptor_pool_ > receptor_pool_cap_) receptor_pool_ = receptor_pool_cap_;
  }

  // optional: for statistics
  inline void add_sum_weight(double w) { sum_weight_ += w; }
  inline double sum_weight() const { return sum_weight_; }
  inline void reset_sum_weight() { sum_weight_ = 0.0; }

  // thresholds & membrane helpers
  inline void set_threshold(double vth) { v_th_ = vth; }
  inline void set_reset(double vreset) { v_reset_ = vreset; }

protected:
  double dt_{1e-3};
  double v_{0.0};
  double v_th_{1.0};
  double v_reset_{0.0};
  double input_current_{0.0};
  bool spiking_{false};

  // receptor resource pool (arbitrary units)
  double receptor_pool_{100.0};
  double receptor_pool_cap_{100.0};

  // stats
  double sum_weight_{0.0};
};

class OutputNeuron : public Neuron {
public:
  using Neuron::Neuron;
  ~OutputNeuron() override = default;
};

// ---------------------------------------------
// Synapses
// ---------------------------------------------

class Synapse {
public:
  Synapse(std::shared_ptr<Neuron> pre,
          std::shared_ptr<Neuron> post,
          double w,
          int flush_interval_sec,
          double dt)
      : pre_neuron_(std::move(pre)),
        post_neuron_(std::move(post)),
        weight_(w),
        flush_interval_sec_(flush_interval_sec),
        dt_(dt) {}
  virtual ~Synapse() = default;

  virtual void transmit(int /*t*/) {
    // simplest current-based synapse: deliver weight as current if pre spikes
    if (pre_neuron_ && pre_neuron_->is_spiking() && post_neuron_) {
      post_neuron_->add_input(weight_);
    }
  }

  virtual void apply_stdp(int /*t*/) = 0; // pure virtual: each synapse defines plasticity
  virtual uint8_t type_code() const = 0;  // 0: AMPA-like, 1: GABA-like

  // common accessors
  inline double weight() const { return weight_; }
  inline void set_weight(double w) { weight_ = w; }

protected:
  std::shared_ptr<Neuron> pre_neuron_;
  std::shared_ptr<Neuron> post_neuron_;
  double weight_{0.0};
  int flush_interval_sec_{0};
  double dt_{1e-3};
};

// --- Excitatory AMPA-like synapse (baseline STDP: no-op by default) ---
class ExcitatorySynapse : public Synapse {
public:
  ExcitatorySynapse(std::shared_ptr<Neuron> pre,
                    std::shared_ptr<Neuron> post,
                    double w,
                    int flush_interval_sec,
                    double dt,
                    double A_plus, double A_minus,
                    double tau_plus, double tau_minus)
      : Synapse(std::move(pre), std::move(post), w, flush_interval_sec, dt),
        A_plus_(A_plus), A_minus_(A_minus),
        tau_plus_(tau_plus), tau_minus_(tau_minus) {}

  ~ExcitatorySynapse() override = default;

  void apply_stdp(int /*t*/) override {
    // Baseline: pure STDP is disabled here to avoid unintended learning
    // (Output uses RExcitatorySynapse; hidden layers may override if needed)
    // No-op.
  }

  uint8_t type_code() const override { return 0; } // AMPA

protected:
  // pair-based STDP parameters (kept for compatibility)
  double A_plus_{0.0}, A_minus_{0.0};
  double tau_plus_{0.02}, tau_minus_{0.02};
};

// --- Inhibitory GABA-like synapse (can have its own STDP rule if desired) ---
class InhibitorySynapse : public Synapse {
public:
  InhibitorySynapse(std::shared_ptr<Neuron> pre,
                    std::shared_ptr<Neuron> post,
                    double w,
                    int flush_interval_sec,
                    double dt)
      : Synapse(std::move(pre), std::move(post), w, flush_interval_sec, dt) {}

  ~InhibitorySynapse() override = default;

  void transmit(int /*t*/) override {
    // inhibitory: subtract current if pre spikes
    if (pre_neuron_ && pre_neuron_->is_spiking() && post_neuron_) {
      post_neuron_->add_input(-std::abs(weight_));
    }
  }

  void apply_stdp(int /*t*/) override {
    // placeholder inhibitory plasticity (no-op by default)
  }

  uint8_t type_code() const override { return 1; } // GABA
};

// --- Reward-modulated AMPA for Output neurons ---
class RExcitatorySynapse : public ExcitatorySynapse {
public:
  using ExcitatorySynapse::ExcitatorySynapse;

  // TD error / reward-prediction error setter
  inline void set_delta(double d) { delta_ = d; }

  // R-STDP hyper-parameters
  inline void set_rstdp_params(double eta, double tau_pre, double tau_post, double tau_elig) {
    eta_ = eta; tau_pre_ = tau_pre; tau_post_ = tau_post; tau_elig_ = tau_elig;
  }

  // Override: "STDP phase" is where R-STDP is actually executed for Output synapses
  void apply_stdp(int /*t*/) override;

  // For compatibility with tooling: keep AMPA code
  uint8_t type_code() const override { return 0; }

private:
  // traces & eligibility
  double pre_trace_{0.0};
  double post_trace_{0.0};
  double elig_{0.0};

  // parameters
  double eta_{1e-3};
  double tau_pre_{0.02};
  double tau_post_{0.02};
  double tau_elig_{0.10};

  // modulatory signal (TD error)
  double delta_{0.0};
};

#endif // NEURON_HPP_
