
from datasets import load_dataset, DownloadConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import aiohttp
from evaluate import load


model_name = 'openai/whisper-large-v3' # raw model: 0.97973

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
# dataset
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
    from tqdm import tqdm
    predictions = []
    references = []
    for batch in tqdm(librispeech_test_clean):
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        reference = processor.tokenizer._normalize(batch['text'])
        references.append(reference)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)[0]
        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)
        predictions.append(prediction)

    # metric
    wer = load("wer")
    wer_result = wer.compute(references=references, predictions=predictions)
    print(f"Result wer: {wer_result * 100}")
    accuracy = 1 - wer_result
    print("Accuracy: %.5f" % accuracy)
    return accuracy


eval_func(model)

