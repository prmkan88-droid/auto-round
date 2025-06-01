import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor, pipeline

from auto_round import AutoRoundMLLM, AutoRound

model_name = "openai/whisper-large-v3"
bits, group_size, sym, act_bits = 8, -1, True, 8


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

## quantize the model
autoround = AutoRoundMLLM(model, tokenizer, processor,
                        bits=bits, group_size=group_size, sym=sym, act_bits=act_bits,
                        iters=0, 
                        layer_config={
                            "proj_out": {
                                "bits": bits, 
                                "group_size": group_size, 
                                "sym": sym, 
                                "act_bits": act_bits,
                            }
                        },
                    )
autoround.quantize()
# print(autoround.model)
# breakpoint()

# save the quantized model, set format='auto_gptq' or 'auto_awq' to use other formats
output_dir = "./rtn_whisper_large_v3"
autoround.save_quantized(output_dir, format='llmcompressor', inplace=True)
# print(autoround.model)
