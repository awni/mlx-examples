import mlx.core as mx
from transformers import AutoConfig, AutoTokenizer
import torch
import transformers
from t5 import T5

hf_path = "DeepFloyd/t5-v1_1-xxl"

def load_t5(hf_path):
    from transformers import T5EncoderModel
    config = AutoConfig.from_pretrained(hf_path).to_dict()
    model = T5EncoderModel.from_pretrained(hf_path, torch_dtype="auto")
    replacements = [
        (".layer.0.layer_norm.", ".ln1."),
        (".layer.1.layer_norm.", ".ln2."),
        (
            "block.0.layer.0.SelfAttention.relative_attention_bias.",
            "relative_attention_bias.embeddings.",
        ),
        (".layer.0.SelfAttention.", ".attention."),
        (".layer.1.DenseReluDense.", ".dense."),
    ]
    def replace(k):
        for o, n in replacements:
            k = k.replace(o, n)
        return k

    weights = [
        (replace(k), mx.array(v)) for k, v in model.state_dict().items()
    ]
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    model = T5(**config)
    model.load_weights(weights)
    return model, tokenizer, config

model, tokenizer, _ = load_t5(hf_path)
text = "a beautiful waterfall"
from opensora.models.text_encoder import T5Encoder
pt_model = T5Encoder(from_pretrained=hf_path, device="mps")
pt_out = pt_model.encode(text)
print(pt_out)

text_tokens_and_mask = tokenizer(
    text,
    max_length=120,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="mlx",
)
out = model.encode(text_tokens_and_mask["input_ids"])
print(out)
