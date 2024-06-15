import torch_griffin
import torch
import numpy as np

import griffin
import common
import mlx.core as mx

config = common.GriffinConfig(
    vocab_size=128,
    width=256,
    mlp_expanded_width=512,
    num_heads=8,
    block_types=(
        common.TemporalBlockType.RECURRENT,
        common.TemporalBlockType.ATTENTION,
    ),
    embeddings_scale_by_sqrt_dim=True,
    attention_window_size=16,
    logits_soft_cap=30.0,
    scan_type=common.ScanType.LINEAR_NATIVE,
)

inputs = np.array([[1, 3, 5, 7, 9]])
segment_pos = np.array([[0, 1, 2, 3, 4]])
torch_model = torch_griffin.Griffin(config=config, dtype=torch.float32)
torch_out = torch_model.forward(torch.tensor(inputs), torch.tensor(segment_pos))
params = torch_model.state_dict()

config = griffin.ModelArgs(
  attention_bias=False,
  conv1d_width=4,
  embeddings_scale_by_sqrt_dim=True,
  hidden_size=256,
  intermediate_size=512,
  logits_soft_cap=30.0,
  attention_window_size=16,
  model_type="recurrent_gemma",
  num_attention_heads=8,
  num_hidden_layers=2,
  num_key_value_heads=1,
  partial_rotary_factor=0.5,
  rms_norm_eps=1e-06,
  rope_theta=10000.0,
  vocab_size=128,
  _block_types=["recurrent", "attention"],
)

params = {k: mx.array(v.detach().numpy()) for k, v in params.items()}

def remap(key):
    key = key.replace("scale", "weight")
    key = key.replace("embedder.input_embedding", "embed_tokens.weight")
    key = key.replace("mlp_block.ffw_down", "mlp_block.down_proj")
    key = key.replace("attention_block", "temporal_block")
    key = key.replace("recurrent_block", "temporal_block")
    key = key.replace("rg_lru.a_gate.b", "rg_lru.recurrent_gate_bias")
    key = key.replace("rg_lru.a_gate.w", "rg_lru.recurrent_gate_weight")
    key = key.replace("rg_lru.input_gate.w", "rg_lru.input_gate_weight")
    key = key.replace("rg_lru.input_gate.b", "rg_lru.input_gate_bias")
    key = key.replace("rg_lru.a_param", "rg_lru.recurrent_param")
    key = key.replace("conv_1d.w", "conv_1d.weight")
    key = key.replace("conv_1d.b", "conv_1d.bias")
    key = key.replace("proj_q", "q_proj")
    key = key.replace("proj_k", "k_proj")
    key = key.replace("proj_v", "v_proj")
    key = key.replace("proj_final", "o_proj")
    return "model." + key

new_params = {}
for k, v in params.items():
    if "mlp_block.ffw_up.w" in k:
        v1, v2 = v.split(2)
        new_params["model." + k.replace("mlp_block.ffw_up.w", "mlp_block.gate_proj.weight")] = v1.squeeze().T
        new_params["model." + k.replace("mlp_block.ffw_up.w", "mlp_block.up_proj.weight")] = v2.squeeze().T
    elif "mlp_block.ffw_up.b" in k:
        v1, v2 = v.split(2)
        new_params["model." + k.replace("mlp_block.ffw_up.b", "mlp_block.gate_proj.bias")] = v1.squeeze()
        new_params["model." + k.replace("mlp_block.ffw_up.b", "mlp_block.up_proj.bias")] = v2.squeeze()


    else:
        new_params[remap(k)] = v

params = new_params

model = griffin.Griffin(config=config)
model.load_weights(list(params.items()))
outputs = model(mx.array(inputs))
print((outputs - torch_out[0].detach().numpy()).abs().max())
import pdb
pdb.set_trace()
