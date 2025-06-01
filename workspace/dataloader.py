import torch


class Dataloader():
    def __init__(self, processor, model, n_samples=2):
        self.batch_size = 1
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.wav_files = self.get_wav_files("/home/sdp/auto-round/workspace/dev-other-prepacked")
        if n_samples is not None:
            self.n_samples = n_samples if n_samples <= len(self.wav_files) else len(self.wav_files)
            if self.n_samples > len(self.wav_files):
                raise ValueError(f"Requested {n_samples} samples, but only {len(self.wav_files)} available.")
        else:
            self.n_samples = len(self.wav_files)
        self.processor = processor
        self.decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        self.inputs = []
        for i in range(self.n_samples):
            self.inputs.append(self.preprocess(self.wav_files[i]))


    def get_wav_files(self, folder_path):
        import os
        wav_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    def __len__(self):
        return self.n_samples

    def preprocess(self, audio_path):
        import librosa
        import numpy as np
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )

        if inputs.input_features.shape[-1] < 3000:
            # ValueError: Whisper expects the mel input features to be of length 3000, but found 585. Make sure to pad the input mel features to 3000.
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                return_attention_mask=True,
            )
        inputs = inputs.to(self.device, dtype=self.torch_dtype)
        return inputs.input_features


    def __iter__(self):
        for i in range(self.n_samples):
            yield {"input_features": self.inputs[i], "decoder_input_ids": self.decoder_input_ids}

