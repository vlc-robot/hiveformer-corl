from typing import List, Union
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, CLIPTextModel, CLIPTokenizer
import torch


def record_mic(duration: float, fs: int) -> np.ndarray:
    import sounddevice as sd

    return sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float64")


class MicToText:
    def __init__(
        self, duration: float = 15.0, sampling_rate: int = 16000, device: str = "cpu"
    ):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = self.model.to(device).eval()
        self.duration = duration
        self.sampling_rate = sampling_rate

    def __call__(self) -> str:
        print("Recording starts:")
        record = record_mic(self.duration, self.sampling_rate)[:, 0]

        # audio file is decoded on the fly
        inputs = self.processor(
            record, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # transcribe speech
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0]


class TextToToken:
    def __init__(self, model_max_length: int = 42, device: str = "cpu"):
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(device).eval()
        self.device = device
        self.model_max_length = model_max_length
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer.model_max_length = model_max_length

    def __call__(self, instruction: Union[str, List[str]]) -> torch.Tensor:
        instructions = instruction if isinstance(instruction, list) else [instruction]
        tokens = self.tokenizer(instructions, padding="max_length")["input_ids"]
        lengths = [len(t) for t in tokens]
        if any(l > self.model_max_length for l in lengths):
            raise RuntimeError(f"Too long instructions: {lengths}")

        tokens = torch.tensor(tokens).to(self.device)
        with torch.no_grad():
            pred = self.model(tokens).last_hidden_state

        return pred


if __name__ == "__main__":
    mic_to_text = MicToText()
    text_to_token = TextToToken()

    instructions = mic_to_text()
    # instructions = "push the rose button, then the orange button, then the black button"
    embeddings = text_to_token(instructions)
    print(embeddings.shape)
