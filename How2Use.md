0) Quick example  

// ---------- 1) Templates ----------  
SynInput = {"max_receptor":10, "flush_interval_sec":100, "dt":0.001}  
SynE     = {"weight":0.5, "flush_interval_sec":100, "dt":0.001, "A_plus":0.01, "A_minus":0.012, "tau_plus":20, "tau_minus":20}  
SynI     = {"weight":-0.5,"flush_interval_sec":100, "dt":0.001, "A_plus":0.00, "A_minus":0.00,  "tau_plus":20, "tau_minus":20}  

// ---------- 2) Layers (GroupAdd) ----------  
L0 += 49  [type=InputNeuron,   setting=SynInput]  
L1 += 784 [type=SpikingNeuron, setting=SynE]  
L2 += 9   [type=OutputNeuron,  setting=SynE]  

// ---------- 3) Edges ----------  
// Layer → Layer (full)  
L0 -> L1 [type=AMPA, mode=full, setting=SynE]  

// Layer → Layer (random): per-post sampling (preferred when both are given)  
L0 -> L1 [type=GABA, mode=random, n_per_post=4, setting=SynI]  

// Layer → Layer (random): per-pre sampling (used only if n_per_post is absent)  
L0 -> L1 [type=AMPA, mode=random, n_to_post=3, setting=SynE]  

// Layer → Neuron  
L0 -> L1_0 [type=AMPA, mode=full, setting=SynE]  
L0 -> L1_0 [type=GABA, mode=random, n_per_post=8, setting=SynI]  

// Neuron → Layer  
L0_5 -> L1 [type=AMPA, mode=full, setting=SynE]  
L0_5 -> L1 [type=GABA, mode=random, n_to_post=12, setting=SynI]  

// Override a single field of a template  
L0_47 -> L1_0 [type=AMPA, setting=SynE{"weight":5.0}]  

1) File structure & syntax  

Each non-empty line is one of:  

Template definition  
Name = { ...JSON... } — stores a reusable JSON object.  

GroupAdd (layer creation/append)  
LayerName += count [ key=value, key=value, ... ]  

Edge (connection)  
Pre -> Post [ key=value, key=value, ... ]  

Rules

// starts a comment; the rest of the line is ignored.  

Whitespace is stripped (L0+=49[...] equals L0 += 49[...]).  

Numbers are decimal; strings may be quoted (quotes are removed).  

Tokens  
 
Layer: L0, L1, Input, Hidden1 (letters/underscore + digits).  

Neuron: L0_5 (layer token + underscore + zero-based index).  

Indices are zero-based.  

Layer order is defined by first appearance in GroupAdd; reusing a name appends to that layer.  

2) Templates  

Define reusable parameter sets for neurons/synapses:  

SynE = {"weight":0.5,"flush_interval_sec":100,"dt":0.001,"A_plus":0.01,"A_minus":0.012,"tau_plus":20,"tau_minus":20}  

Template override (patch)  

Anywhere setting=SynE is accepted, you can patch it inline:  

setting=SynE{"weight":5.0}  


Semantics: JSON Merge Patch — patched keys replace the base.  

3) GroupAdd (layers & neurons)  
LayerName += count [type=<NeuronClass>, setting=<Template or Template{patch}>]  


NeuronClass & expected fields  

InputNeuron  
max_receptor, flush_interval_sec, dt  

SpikingNeuron / OutputNeuron  
max_receptor, flush_interval_sec, tau_m, v_rest, v_reset, v_th, refractory_period, dt  

Example  

L0 += 49  [type=InputNeuron,   setting=SynInput]  
L1 += 784 [type=SpikingNeuron, setting=SynE]  
L2 += 9   [type=OutputNeuron,  setting=SynE]  

4) Edges (connections)  
Pre -> Post [type=<AMPA|GABA>, mode=<full|random>, setting=<SynTemplate or SynTemplate{patch}>, ...]  


Allowed endpoints  

Layer → Layer: L0 -> L1  

Neuron → Neuron: L0_5 -> L1_0  

Layer → Neuron: L0 -> L1_0  

Neuron → Layer: L0_5 -> L1  

Constraint: the last layer cannot be on the left (pre).  
(Outgoing synapse buckets exist only for layers [0 .. layers-2].)  

type  

AMPA (excitatory), GABA (inhibitory).  
Both may exist between the same (pre, post) pair; re-adding the same (pre, post, type) replaces the previous one.  

mode  

full — connect all implied (pre, post) pairs.  

random — connect a subset (parameters below).  

Random parameters & precedence  
Direction	Parameter(s)	Meaning	Precedence  
Layer → Layer	n_per_post=<k> or n_to_post=<k>	If n_per_post is present, per-post sampling: for each post, choose k distinct pres. Otherwise, per-pre: for each   pre, choose k distinct posts.	n_per_post wins if both are present  
Layer → Neuron	n_per_post=<k>	For the single post neuron, choose k distinct pres.	—  
Neuron → Layer	n_to_post=<k> (or n_per_post)	From the single pre neuron, choose k distinct posts.	— (accepts n_per_post too)  

Examples  

// Layer → Layer (random): per-post sampling (preferred if both are present)  
L0 -> L1 [type=AMPA, mode=random, n_per_post=4, setting=SynE]  

// Layer → Layer (random): per-pre sampling (only used if n_per_post is absent)  
L0 -> L1 [type=GABA, mode=random, n_to_post=3, setting=SynI]  

// Layer → Neuron (random): pick 8 pres for L1_0  
L0 -> L1_0 [type=AMPA, mode=random, n_per_post=8, setting=SynE]  

// Neuron → Layer (random): pre L0_5 connects to 12 posts in L1  
L0_5 -> L1 [type=AMPA, mode=random, n_to_post=12, setting=SynE]  

5) Parameter reference  

Neuron template fields  

max_receptor (double) — receptor pool capacity per neuron  

flush_interval_sec (int) — logging/flush cadence (seconds)  

dt (double seconds) — integration timestep  

For SpikingNeuron / OutputNeuron:  
tau_m, v_rest, v_reset, v_th, refractory_period (doubles)  

Synapse template fields  

weight (double) — initial weight  

flush_interval_sec (int), dt (double)  

STDP: A_plus, A_minus, tau_plus, tau_minus  

Step update order: AMPA → GABA (stable).  
Inhibitory updates may use excitatory statistics computed earlier in the same step.  

6) Errors & diagnostics  

Unknown template: <Name> — setting= references an undefined template  

Not a valid groupadd line: / Bad edge line: — syntax mismatch  

pre/post endpoint out of range — layer or neuron index not found  

last layer cannot be a pre endpoint — attempted to connect from the final layer  

unknown synapse type: / unknown mode: — unsupported values  

7) Useful patterns  

Patch a single field on one edge  

L0_47 -> L1_0 [type=AMPA, setting=SynE{"weight":5.0}]  


AMPA + GABA between the same layers (engine sorts AMPA→GABA)  

L0 -> L1 [type=AMPA, mode=full,   setting=SynE]  
L0 -> L1 [type=GABA, mode=random, n_per_post=2, setting=SynI]  


Classic 3-layer forward net  

In  += 64  [type=InputNeuron,   setting=SynInput]  
Hid += 256 [type=SpikingNeuron, setting=SynE]  
Out += 10  [type=OutputNeuron,  setting=SynE]  

In  -> Hid [type=AMPA, mode=full, setting=SynE]  
Hid -> Out [type=AMPA, mode=full, setting=SynE]  


Layer → Layer random, per-pre sampling (new)

L0 -> L1 [type=AMPA, mode=random, n_to_post=5, setting=SynE]  // used only if n_per_post is absent
