# Whisper quantization

To get the RTN model, please install this branch and run
```python
python test_rtn.py
```

To get the AutoRound fine-tuned model.
```python
python round_whisper.py
```

To test accuracy on librispeech_asr/clean test dataset:
```python
python test_raw_wer.py  # testing using transformers raw model
python test_vllm_wer.py  # testing using vLLM raw model
python test_llmc_wer.py  # testing using llmcompressor quantized model (w8a8 + FP8 kv_cache)
python test_atrd_wer.py  # testing using autoround quantized model (w8a8 + FP8 kv_cache)
```


