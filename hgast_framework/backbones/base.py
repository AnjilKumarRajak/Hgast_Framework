
from abc import ABC, abstractmethod
from typing import Optional


class TranslationBackbone(ABC):
    name: str = "unnamed_backbone"

    @abstractmethod
    def translate_en_to_hi(self, text: str) -> str:
        raise NotImplementedError

    def transcribe(self, audio_array, sampling_rate: int) -> Optional[str]:
        raise NotImplementedError(
            f"{self.name} does not implement transcribe(); "
            f"pass English text directly instead."
        )

    def load(self):
        pass

    def unload(self):
        pass

    def __repr__(self):
        return f"<TranslationBackbone: {self.name}>"
