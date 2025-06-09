from vllm.assets.audio import AudioAsset
from vllm import LLM, SamplingParams

# model_id = "RedHatAI/whisper-large-v3-quantized.w8a8"  # autoround exports the same format
model_id = "./rtn_whisper_large_v3"
# model_id = "./atrd_whisper_large_v3"


# 1. raw: 
# 2. raw+fp8_KV: 

# prepare model
llm = LLM(
    model=model_id,
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",  # fp8_e5m2, -> bf16 gemm  # INT8?
)
model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model.model

# # prepare inputs
# inputs = {  # Test explicit encoder/decoder prompt
#     "encoder_prompt": {
#         "prompt": "",
#         "multi_modal_data": {
#             "audio": AudioAsset("winning_call").audio_and_sample_rate,
#         },
#     },
#     "decoder_prompt": "<|startoftranscript|>",
# }
# # (Pdb) AudioAsset("winning_call").audio_and_sample_rate
# # (array([-0.00061354, -0.00054806, -0.00036311, ...,  0.00974603,
# #         0.01459668,  0.01831684], dtype=float32), 44100)
# # breakpoint()

# # generate response
# print("========== SAMPLE GENERATION ==============")
# outputs = llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=64))
# print(f"PROMPT  : {outputs[0].prompt}")
# print(f"RESPONSE: {outputs[0].outputs[0].text}")
# print("==========================================")


# dataset
import aiohttp
from datasets import load_dataset
librispeech_test_clean = load_dataset(
    "librispeech_asr", 
    "clean", 
    split="test", 
    cache_dir="/dev/shm/librispeech_asr_clean_test",
    # streaming=True,
    trust_remote_code=True,
    storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=60000)}}
)


def eval_func(model):
    from evaluate import load
    from tqdm import tqdm
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(model_id)
    predictions = []
    references = []
    for batch in tqdm(librispeech_test_clean):
        audio = batch["audio"]
        inputs = {  # Test explicit encoder/decoder prompt
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio["array"], audio["sampling_rate"]),
                },
            },
            "decoder_prompt": "<|startoftranscript|>",
        }

        reference = processor.tokenizer._normalize(batch['text'])
        references.append(reference)
        outputs = llm.generate(
            inputs, 
            SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=448,
            ), 
            use_tqdm=False
        )
        prediction = outputs[0].outputs[0].text
        prediction = processor.tokenizer._normalize(prediction)
        predictions.append(prediction)

    # metric
    wer = load("wer")
    wer_result = wer.compute(references=references, predictions=predictions)
    print(f"Result wer: {wer_result * 100}")
    accuracy = 1 - wer_result
    print("Accuracy: %.5f" % accuracy)
    return accuracy


eval_func(model)


