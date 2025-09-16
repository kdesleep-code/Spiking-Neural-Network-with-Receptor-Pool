/**
 * @file SpikingNeuralNetwork.cpp
 * @brief SNN全体の初期化・接続生成・実行・出力の実装
 *
 * @details
 * - 構成ファイル（DSL）を読み込み、テンプレート適用・グループ追加・エッジ追加を行う。
 * - ステップ処理は Neuron -> Synapse -> STDP（AMPA → GABA）の順で進行。
 * - synapses は常に「layers-1」に揃え、最終層を pre にしない前提で安全側に。
 */

#include "SpikingNeuralNetwork.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <numeric>  // std::iota

#include "./static_lib/json.hpp"
using json = nlohmann::json;

// ==================== 既存の内部ヘルパ（匿名名前空間） ====================
namespace {

std::string sanitize_line(std::string s) {
  auto p = s.find("//");
  if (p != std::string::npos) s = s.substr(0, p);
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  s.erase(std::remove_if(s.begin(), s.end(),
                         [](char c) { return c == ' ' || c == '\t'; }),
          s.end());
  return s;
}

std::vector<std::string> split_top_level_commas(const std::string& s) {
  std::vector<std::string> parts;
  std::string cur;
  int brace = 0;
  bool in_quote = false;
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '"' && (i == 0 || s[i - 1] != '\\')) in_quote = !in_quote;
    if (!in_quote) {
      if (c == '{') ++brace;
      else if (c == '}') --brace;
      else if (c == ',' && brace == 0) {
        parts.push_back(cur);
        cur.clear();
        continue;
      }
    }
    cur.push_back(c);
  }
  if (!cur.empty()) parts.push_back(cur);
  return parts;
}

std::string unquote_if_needed(std::string v) {
  if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
    v = v.substr(1, v.size() - 2);
  }
  return v;
}

std::unordered_map<std::string, std::string> parse_bracket_params(
    const std::string& inside) {
  std::unordered_map<std::string, std::string> kv;
  for (auto& p : split_top_level_commas(inside)) {
    auto eq = p.find('=');
    if (eq == std::string::npos) continue;
    std::string k = p.substr(0, eq);
    std::string v = p.substr(eq + 1);
    kv[k] = unquote_if_needed(v);
  }
  return kv;
}

json effective_json_from_setting(
    const std::string& setting_str,
    const std::unordered_map<std::string, std::string>& template_map) {
  std::smatch m;
  static const std::regex re_override(R"(^(\w+)(\{.*\})$)");
  if (std::regex_match(setting_str, m, re_override)) {
    const std::string base_name = m[1];
    const std::string patch_str = m[2];
    auto it = template_map.find(base_name);
    if (it == template_map.end())
      throw std::runtime_error("Unknown template: " + base_name);
    json base = json::parse("{" + it->second + "}");
    json patch = json::parse(patch_str);
    base.merge_patch(patch);
    return base;
  } else {
    auto it = template_map.find(setting_str);
    if (it == template_map.end())
      throw std::runtime_error("Unknown template: " + setting_str);
    return json::parse("{" + it->second + "}");
  }
}

struct Endpoint {
  int layer_id = -1;
  bool is_layer = true;  // true: Lx, false: Lx_i
  int neuron_id = -1;    // is_layer == false で有効
};

bool parse_endpoint_token(const std::string& token,
                          const std::unordered_map<std::string, int>&
                              layer_name_to_id,
                          Endpoint& out) {
  std::string lname = token;
  int idx = -1;

  const auto us = token.rfind('_');
  if (us != std::string::npos && us + 1 < token.size()) {
    const std::string tail = token.substr(us + 1);
    const bool all_digits =
        std::all_of(tail.begin(), tail.end(),
                    [](unsigned char c) { return std::isdigit(c); });
    if (all_digits) {
      lname = token.substr(0, us);
      idx = std::stoi(tail);
    }
  }

  auto it = layer_name_to_id.find(lname);
  if (it == layer_name_to_id.end()) return false;

  out.layer_id = it->second;
  if (idx >= 0) {
    out.is_layer = false;
    out.neuron_id = idx;
  } else {
    out.is_layer = true;
    out.neuron_id = -1;
  }
  return true;
}

} // anonymous namespace
// ==================== 内部ヘルパここまで ====================

SpikingNeuralNetwork::SpikingNeuralNetwork(uint32_t seed) {
  if (seed == 0) {
    std::random_device rd;
    seed_ = rd();
  } else {
    seed_ = seed;
  }
  rng_ = std::mt19937(seed_);
  current_step = 0;
}

// =============== export_graph_bin（現行どおり） ===============
void SpikingNeuralNetwork::export_graph_bin(const std::string& filename) const {
  std::ofstream ofs(filename, std::ios::binary);

  // ノード部
  int32_t node_count = 0;
  for (const auto& layer : neurons)
    node_count += static_cast<int32_t>(layer.size());
  ofs.write(reinterpret_cast<const char*>(&node_count), sizeof(int32_t));
  for (size_t l = 0; l < neurons.size(); ++l) {
    for (size_t n = 0; n < neurons[l].size(); ++n) {
      int32_t layer_id = static_cast<int32_t>(l);
      int32_t neuron_id = static_cast<int32_t>(n);
      uint8_t type = neurons[l][n]->type_code(); // 仮想関数で型取得
      ofs.write(reinterpret_cast<const char*>(&layer_id), sizeof(int32_t));
      ofs.write(reinterpret_cast<const char*>(&neuron_id), sizeof(int32_t));
      ofs.write(reinterpret_cast<const char*>(&type), sizeof(uint8_t));
    }
  }

  // エッジ部
  int32_t edge_count = 0;
  for (const auto& synvec : synapses)
    edge_count += static_cast<int32_t>(synvec.size());
  ofs.write(reinterpret_cast<const char*>(&edge_count), sizeof(int32_t));
  for (const auto& synvec : synapses) {
    for (const auto& syn : synvec) {
      auto pre = syn->get_pre_neuron();
      auto post = syn->get_post_neuron();
      int32_t pre_layer = pre->get_layer_id();
      int32_t pre_id = pre->get_neuron_id();
      int32_t post_layer = post->get_layer_id();
      int32_t post_id = post->get_neuron_id();
      uint8_t type = syn->type_code(); // 0(AMPA)→1(GABA)

      ofs.write(reinterpret_cast<const char*>(&pre_layer), sizeof(int32_t));
      ofs.write(reinterpret_cast<const char*>(&pre_id), sizeof(int32_t));
      ofs.write(reinterpret_cast<const char*>(&post_layer), sizeof(int32_t));
      ofs.write(reinterpret_cast<const char*>(&post_id), sizeof(int32_t));
      ofs.write(reinterpret_cast<const char*>(&type), sizeof(uint8_t));
    }
  }
  ofs.close();
}

// =============== AMPA→GABA の安定ソート（現行仕様のまま） ===============
void SpikingNeuralNetwork::sort_synapses_by_type() {
  for (auto& syn_vec : synapses) {
    std::stable_sort(
      syn_vec.begin(), syn_vec.end(),
      [](const std::shared_ptr<Synapse>& a, const std::shared_ptr<Synapse>& b){
        return a->type_code() < b->type_code(); // 0(AMPA)→1(GABA)
      });
  }
}

// =============== init()：段階ヘルパに分割 & 堅牢化 ===============
void SpikingNeuralNetwork::init(const std::string& filename) {
  std::vector<std::vector<std::string>> buckets;                    // [Template/Group/Edge/Other]
  std::unordered_map<std::string, std::string> template_map;        // name -> json string inside {...}
  std::unordered_map<std::string, int> layer_name_to_id;            // "L0" 等の名前 -> layer_id

  parse_and_bucket_lines(filename, buckets);
  build_template_map(buckets.size() > 0 ? buckets[0] : std::vector<std::string>{}, template_map);
  apply_group_add(buckets.size() > 1 ? buckets[1] : std::vector<std::string>{}, template_map, layer_name_to_id);

  // ★ レイヤ数に合わせて synapses を「layers-1」に初期化（最終層は pre にならない）
  const int layers = static_cast<int>(neurons.size());
  synapses.clear();
  synapses.resize(std::max(0, layers - 1));

  apply_edges(buckets.size() > 2 ? buckets[2] : std::vector<std::string>{}, template_map, layer_name_to_id);

  // AMPA→GABA へ並び替え
  sort_synapses_by_type();

  // 念のためもう一度「layers-1」に丸める
  if (static_cast<int>(synapses.size()) != std::max(0, layers - 1)) {
    synapses.resize(std::max(0, layers - 1));
  }

  current_step = 0;
}

// ---- ヘルパ1：行をタイプ別にバケツ分け ----
void SpikingNeuralNetwork::parse_and_bucket_lines(
    const std::string& filename,
    std::vector<std::vector<std::string>>& out_buckets) {
  std::ifstream ifs(filename);
  if (!ifs) {
    throw std::runtime_error("Error: Cannot open config file: " + filename);
  }

  // 0:Template, 1:GroupAdd, 2:Edge, 3:Other
  out_buckets.assign(4, {});
  std::regex re_template(R"(^([A-Za-z_]\w*)=\{(.*)\}$)");
  std::regex re_groupadd(R"(^([A-Za-z_]\w*)\+=([0-9]+)\[(.*)\]$)");
  std::regex re_edge_bucket(R"(^([A-Za-z_]\w*(?:_[0-9]+)?)\->([A-Za-z_]\w*(?:_[0-9]+)?)\[)");

  std::string raw;
  while (std::getline(ifs, raw)) {
    const std::string s = sanitize_line(raw);
    if (s.empty()) continue;

    std::smatch m;
    int idx = 3;
    if (std::regex_search(s, m, re_template))      idx = 0;
    else if (std::regex_search(s, m, re_groupadd)) idx = 1;
    else if (std::regex_search(s, m, re_edge_bucket)) idx = 2;

    out_buckets[idx].push_back(s);
  }
}

// ---- ヘルパ2：Templateマップ構築 ----
void SpikingNeuralNetwork::build_template_map(
    const std::vector<std::string>& template_lines,
    std::unordered_map<std::string, std::string>& template_map) {
  std::regex re_template(R"(^([A-Za-z_]\w*)=\{(.*)\}$)");
  for (const auto& s : template_lines) {
    std::smatch m;
    if (std::regex_match(s, m, re_template)) {
      template_map[m[1]] = m[2];  // {…} の中身
    } else {
      std::cerr << "[WARN] Not a valid template line: " << s << std::endl;
    }
  }
}

// ---- ヘルパ3：GroupAdd適用（Neuron追加） ----
void SpikingNeuralNetwork::apply_group_add(
    const std::vector<std::string>& group_lines,
    const std::unordered_map<std::string, std::string>& template_map,
    std::unordered_map<std::string, int>& layer_name_to_id) {

  std::regex re_groupadd(R"(^([A-Za-z_]\w*)\+=([0-9]+)\[(.*)\]$)");
  int next_layer_id = static_cast<int>(neurons.size()); // 既存があれば後ろに続ける

  for (const auto& s : group_lines) {
    std::smatch m;
    if (!std::regex_match(s, m, re_groupadd)) {
      std::cerr << "[WARN] Not a valid groupadd line: " << s << std::endl;
      continue;
    }
    const std::string layer_name = m[1];
    const int count = std::stoi(m[2]);
    const std::string params_str = m[3];

    // LayerIDの割当/再利用
    int layer_id;
    if (layer_name_to_id.count(layer_name))
      layer_id = layer_name_to_id[layer_name];
    else {
      layer_id = next_layer_id++;
      layer_name_to_id[layer_name] = layer_id;
      if (static_cast<int>(neurons.size()) <= layer_id) neurons.resize(layer_id + 1);
      // synapses はここでは触らない（initで layers-1 に揃える）
    }

    const auto kv = parse_bracket_params(params_str);
    const std::string neuron_type = (kv.count("type") ? kv.at("type") : "");
    const std::string setting_str = (kv.count("setting") ? kv.at("setting") : "");

    if (neuron_type.empty() || setting_str.empty()) {
      std::cerr << "[WARN] Missing type/setting in: " << s << std::endl;
      continue;
    }

    json sj;
    try {
      sj = effective_json_from_setting(setting_str, template_map);
    } catch (const std::exception& e) {
      std::cerr << "[WARN] " << e.what() << " in line: " << s << std::endl;
      continue;
    }

    if (neuron_type == "InputNeuron") {
      add_neurons<InputNeuron>(layer_id, count, sj["max_receptor"],
                               sj["flush_interval_sec"], sj["dt"]);
    } else if (neuron_type == "SpikingNeuron") {
      add_neurons<SpikingNeuron>(layer_id, count, sj["max_receptor"],
                                 sj["flush_interval_sec"], sj["tau_m"], sj["v_rest"],
                                 sj["v_reset"], sj["v_th"], sj["refractory_period"],
                                 sj["dt"]);
    } else if (neuron_type == "OutputNeuron") {
      add_neurons<OutputNeuron>(layer_id, count, sj["max_receptor"],
                                sj["flush_interval_sec"], sj["tau_m"], sj["v_rest"],
                                sj["v_reset"], sj["v_th"], sj["refractory_period"],
                                sj["dt"]);
    } else {
      std::cerr << "[WARN] Unknown neuron type: " << neuron_type << std::endl;
      continue;
    }

    std::cout << "Add " << count << " " << neuron_type << " to " << layer_name
              << " (LayerID=" << layer_id << ")\n";
  }
}

// ---- ヘルパ4：Edge適用（Synapse追加） ----
void SpikingNeuralNetwork::apply_edges(
    const std::vector<std::string>& edge_lines,
    const std::unordered_map<std::string, std::string>& template_map,
    const std::unordered_map<std::string, int>& layer_name_to_id) {

  static const std::regex re_edge_full(
      R"(^([A-Za-z_]\w*(?:_[0-9]+)?)\->([A-Za-z_]\w*(?:_[0-9]+)?)\[(.*)\]$)");

  auto endpoint_in_range = [&](const Endpoint& ep) -> bool {
    if (ep.layer_id < 0 || ep.layer_id >= static_cast<int>(neurons.size()))
      return false;
    if (ep.is_layer) return true;
    if (ep.neuron_id < 0 ||
        ep.neuron_id >= static_cast<int>(neurons[ep.layer_id].size()))
      return false;
    return true;
  };

  const int layers = static_cast<int>(neurons.size());

  for (const auto& s : edge_lines) {
    std::smatch m;
    if (!std::regex_match(s, m, re_edge_full)) {
      std::cerr << "[WARN] Bad edge line: " << s << "\n";
      continue;
    }
    const std::string pre_tok = m[1];
    const std::string post_tok = m[2];
    const std::string inside = m[3];

    Endpoint pre_ep, post_ep;
    if (!parse_endpoint_token(pre_tok, layer_name_to_id, pre_ep) ||
        !parse_endpoint_token(post_tok, layer_name_to_id, post_ep)) {
      std::cerr << "[WARN] Unknown layer/neuron token: " << s << "\n";
      continue;
    }

    if (!endpoint_in_range(pre_ep)) {
      std::cerr << "[ERR] pre endpoint out of range: " << pre_tok << "\n";
      continue;
    }
    if (!endpoint_in_range(post_ep)) {
      std::cerr << "[ERR] post endpoint out of range: " << post_tok << "\n";
      continue;
    }

    // ★ 最終層は pre になれない
    if (pre_ep.layer_id >= layers - 1) {
      std::cerr << "[WARN] last layer cannot be a pre endpoint: " << s << "\n";
      continue;
    }

    const auto kv = parse_bracket_params(inside);
    const std::string type = kv.count("type") ? kv.at("type") : "AMPA";
    const std::string mode = kv.count("mode") ? kv.at("mode") : "full";
    if (!kv.count("setting")) {
      std::cerr << "[WARN] missing setting: " << s << "\n";
      continue;
    }

    json sj;
    try {
      sj = effective_json_from_setting(kv.at("setting"), template_map);
    } catch (const std::exception& e) {
      std::cerr << "[WARN] " << e.what() << " in line: " << s << "\n";
      continue;
    }

    auto add_one_syn = [&](bool excit, int pre_layer_id, int pre_neuron_id, int post_layer_id, int post_neuron_id) {
      if (excit) {
        if (remove_synapse_if_exists<ExcitatorySynapse>(pre_layer_id, pre_neuron_id, post_layer_id, post_neuron_id)) {
          std::cout << "[INFO] replaced AMPA " << pre_layer_id << "_" << pre_neuron_id << "->" << post_layer_id
                    << "_" << post_neuron_id << "\n";
        }
        add_synapse<ExcitatorySynapse>(
            pre_layer_id, pre_neuron_id, post_layer_id, post_neuron_id, sj["weight"], sj["flush_interval_sec"], sj["dt"],
            sj["A_plus"], sj["A_minus"], sj["tau_plus"], sj["tau_minus"]);
      } else {
        if (remove_synapse_if_exists<InhibitorySynapse>(pre_layer_id, pre_neuron_id, post_layer_id, post_neuron_id)) {
          std::cout << "[INFO] replaced GABA " << pre_layer_id << "_" << pre_neuron_id << "->" << post_layer_id
                    << "_" << post_neuron_id << "\n";
        }
        add_synapse<InhibitorySynapse>(
            pre_layer_id, pre_neuron_id, post_layer_id, post_neuron_id, sj["weight"], sj["flush_interval_sec"], sj["dt"],
            sj["A_plus"], sj["A_minus"], sj["tau_plus"], sj["tau_minus"]);
      }
    };

    const bool is_excit = (type == "AMPA");
    if (!(is_excit || type == "GABA")) {
      std::cerr << "[WARN] unknown synapse type: " << type << " in " << s << "\n";
      continue;
    }

    // Layer->Layer
    if (pre_ep.is_layer && post_ep.is_layer) {
      const int n_pre  = static_cast<int>(neurons[pre_ep.layer_id].size());
      const int n_post = static_cast<int>(neurons[post_ep.layer_id].size());
      if (n_pre <= 0 || n_post <= 0) continue;

      if (mode == "full") {
        if (is_excit) {
          add_full_connect<ExcitatorySynapse>(
              pre_ep.layer_id, post_ep.layer_id, sj["weight"],
              sj["flush_interval_sec"], sj["dt"], sj["A_plus"], sj["A_minus"],
              sj["tau_plus"], sj["tau_minus"]);
        } else {
          add_full_connect<InhibitorySynapse>(
              pre_ep.layer_id, post_ep.layer_id, sj["weight"],
              sj["flush_interval_sec"], sj["dt"], sj["A_plus"], sj["A_minus"],
              sj["tau_plus"], sj["tau_minus"]);
        }
      } else if (mode == "random") {
        const bool has_npp = kv.count("n_per_post") > 0;
        const bool has_ntp = kv.count("n_to_post")  > 0;

        if (has_npp || !has_ntp) {
          // 既存の意味: 各 post について pre を n_per_post 本選ぶ（優先）
          int npp = 1;
          if (has_npp) npp = std::stoi(kv.at("n_per_post"));
          npp = std::max(0, npp);
          if (is_excit) {
            add_random_connect<ExcitatorySynapse>(
                pre_ep.layer_id, post_ep.layer_id, npp, sj["weight"],
                sj["flush_interval_sec"], sj["dt"], sj["A_plus"], sj["A_minus"],
                sj["tau_plus"], sj["tau_minus"]);
          } else {
            add_random_connect<InhibitorySynapse>(
                pre_ep.layer_id, post_ep.layer_id, npp, sj["weight"],
                sj["flush_interval_sec"], sj["dt"], sj["A_plus"], sj["A_minus"],
                sj["tau_plus"], sj["tau_minus"]);
          }
        } else {
          // 新規の意味: 各 pre について post を n_to_post 本選ぶ（n_per_post 未指定時のみ）
          int ntp = std::stoi(kv.at("n_to_post"));
          ntp = std::max(0, ntp);
          std::vector<int> idx(n_post);
          std::iota(idx.begin(), idx.end(), 0);
          const int k = std::min(ntp, n_post);
          for (int i = 0; i < n_pre; ++i) {
            std::shuffle(idx.begin(), idx.end(), rng_);
            for (int t = 0; t < k; ++t) {
              add_one_syn(is_excit, pre_ep.layer_id, i, post_ep.layer_id, idx[t]);
            }
          }
        }
      } else {
        std::cerr << "[WARN] unknown mode: " << mode << " in " << s << "\n";
        continue;
      }
    }
    // Neuron->Neuron
    else if (!pre_ep.is_layer && !post_ep.is_layer) {
      add_one_syn(is_excit, pre_ep.layer_id, pre_ep.neuron_id, post_ep.layer_id,
                  post_ep.neuron_id);
    }
    // Layer->Neuron
    else if (pre_ep.is_layer && !post_ep.is_layer) {
      if (pre_ep.layer_id >= layers - 1) {
        std::cerr << "[WARN] last layer cannot be a pre endpoint: " << s << "\n";
        continue;
      }
      const int n_pre = static_cast<int>(neurons[pre_ep.layer_id].size());
      if (n_pre <= 0) continue;

      if (mode == "full") {
        for (int i = 0; i < n_pre; ++i) {
          add_one_syn(is_excit, pre_ep.layer_id, i, post_ep.layer_id, post_ep.neuron_id);
        }
      } else if (mode == "random") {
        int pick = 1;
        if (kv.count("n_per_post")) pick = std::stoi(kv.at("n_per_post"));
        pick = std::max(0, pick);
        std::vector<int> idx(n_pre);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng_);
        const int k = std::min(pick, n_pre);
        for (int t = 0; t < k; ++t) {
          add_one_syn(is_excit, pre_ep.layer_id, idx[t], post_ep.layer_id, post_ep.neuron_id);
        }
      } else {
        std::cerr << "[WARN] unknown mode: " << mode << "\n";
        continue;
      }
    }
    // Neuron->Layer
    else if (!pre_ep.is_layer && post_ep.is_layer) {
      if (pre_ep.layer_id >= layers - 1) {
        std::cerr << "[WARN] last layer cannot be a pre endpoint: " << s << "\n";
        continue;
      }
      const int n_post = static_cast<int>(neurons[post_ep.layer_id].size());
      if (n_post <= 0) continue;

      if (mode == "full") {
        for (int j = 0; j < n_post; ++j) {
          add_one_syn(is_excit, pre_ep.layer_id, pre_ep.neuron_id, post_ep.layer_id, j);
        }
      } else if (mode == "random") {
        int pick = 1;
        if (kv.count("n_to_post"))       pick = std::stoi(kv.at("n_to_post"));
        else if (kv.count("n_per_post")) pick = std::stoi(kv.at("n_per_post")); // 互換
        pick = std::max(0, pick);

        std::vector<int> idx(n_post);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng_);
        const int k = std::min(pick, n_post);
        for (int t = 0; t < k; ++t) {
          add_one_syn(is_excit, pre_ep.layer_id, pre_ep.neuron_id, post_ep.layer_id, idx[t]);
        }
      } else {
        std::cerr << "[WARN] unknown mode: " << mode << "\n";
        continue;
      }
    }
  }
}

// =============== ラン/ステップ系（現行どおり） ===============
void SpikingNeuralNetwork::set_inputs(const std::vector<double>& values) {
  size_t idx = 0;
  for (auto& layer : neurons) {
    for (auto& n_ptr : layer) {
      if (auto* inp = dynamic_cast<InputNeuron*>(n_ptr.get())) {
        if (idx < values.size()) {
          inp->set_input(values[idx]);
          ++idx;
        } else {
          inp->set_input(0.0);
        }
      }
    }
  }
}

void SpikingNeuralNetwork::step() {
  const size_t L = neurons.size();
  // 1. 各層のニューロンを進める
  for (size_t l = 0; l < L; ++l) {
    for (auto& n : neurons[l]) {
      if (n) n->step(current_step);
    }
    // 2. その層から出るシナプスを伝達
    if (l < synapses.size()) {
      for (auto& s : synapses[l]) {
        if (s) s->transmit(current_step);
      }
    }
  }
  // 3. STDP（AMPA→GABAの順）
  for (size_t l = 0; l < synapses.size(); ++l) {
    for (auto& s : synapses[l]) {
      if (s) s->apply_stdp(current_step);
    }
  }
  ++current_step;
}

void SpikingNeuralNetwork::run(int steps) {
  for (int i = 0; i < steps; ++i) step();
}

// =============== STDP Freeze（現行APIのまま） ===============
void SpikingNeuralNetwork::disableSTDP(int layer_id, int neuron_id) {
  neurons[layer_id][neuron_id]->set_freeze(true);
}
void SpikingNeuralNetwork::disableSTDP(int layer_id) {
  for (auto& n : neurons[layer_id]) n->set_freeze(true);
}
void SpikingNeuralNetwork::enableSTDP(int layer_id, int neuron_id) {
  neurons[layer_id][neuron_id]->set_freeze(false);
}
void SpikingNeuralNetwork::enableSTDP(int layer_id) {
  for (auto& n : neurons[layer_id]) n->set_freeze(false);
}

// =============== Sleep モード一括制御（追加） ===============
void SpikingNeuralNetwork::set_sleep_mode_all_spiking(bool on, double rate_hz) {
  for (auto& layer : neurons) {
    for (auto& n : layer) {
      if (auto* spk = dynamic_cast<SpikingNeuron*>(n.get())) {
        spk->set_sleep_mode(on, rate_hz);
      }
    }
  }
}

void SpikingNeuralNetwork::set_sleep_mode_on_layer(int layer_id, bool on, double rate_hz) {
  if (layer_id < 0 || layer_id >= static_cast<int>(neurons.size())) return;
  for (auto& n : neurons[layer_id]) {
    if (auto* spk = dynamic_cast<SpikingNeuron*>(n.get())) {
      spk->set_sleep_mode(on, rate_hz);
    }
  }
}

void SpikingNeuralNetwork::set_sleep_mode_on_neuron(int layer_id, int neuron_id, bool on, double rate_hz) {
  if (layer_id < 0 || layer_id >= static_cast<int>(neurons.size())) return;
  if (neuron_id < 0 || neuron_id >= static_cast<int>(neurons[layer_id].size())) return;
  if (auto* spk = dynamic_cast<SpikingNeuron*>(neurons[layer_id][neuron_id].get())) {
    spk->set_sleep_mode(on, rate_hz);
  }
}
