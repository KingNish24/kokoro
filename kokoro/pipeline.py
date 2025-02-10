from .model import KModel
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from misaki import en, espeak
from numbers import Number
from typing import Generator, List, Optional, Tuple, Union
import re
import torch

ALIASES = {
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'pt-br': 'p',
    'ja': 'j',
    'zh': 'z',
}

LANG_CODES = dict(
    # pip install misaki[en]
    a='American English',
    b='British English',

    # espeak-ng
    e='es',
    f='fr-fr',
    h='hi',
    i='it',
    p='pt-br',

    # pip install misaki[ja]
    j='Japanese',

    # pip install misaki[zh]
    z='Mandarin Chinese',
)

class AutoregressiveStreamKPipeline:
    """
    AutoregressiveStreamKPipeline attempts sample-by-sample streaming TTS.

    IMPORTANT CAVEATS:

    *   This pipeline is HIGHLY EXPERIMENTAL and likely to have significant
        quality degradation compared to the chunking or full-sentence versions.
    *   The underlying KModel is NOT designed for true autoregressive,
        sample-by-sample generation. This pipeline forces it to behave that
        way, leading to potential artifacts and instability.
    *   Efficiency: While this aims for streaming, the per-phoneme processing
        overhead will make it SLOWER than the chunking version.
    * No Timestamp Support
    """

    def __init__(
        self,
        lang_code: str,
        model: Union[KModel, bool] = True,
        trf: bool = False,
        device: Optional[str] = None,
    ):
        lang_code = lang_code.lower()
        lang_code = ALIASES.get(lang_code, lang_code)
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        self.model = None
        if isinstance(model, KModel):
            self.model = model
        elif model:
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self.model = KModel().to(device).eval()
            except RuntimeError as e:
                if device == 'cuda':
                    raise RuntimeError(f"""Failed to initialize model on CUDA: {e}.
                                       Try setting device='cpu' or check CUDA installation.""")
                raise
        self.voices = {}
        if lang_code in 'ab':
            try:
                fallback = espeak.EspeakFallback(british=lang_code=='b')
            except Exception as e:
                logger.warning("EspeakFallback not Enabled: OOD words will be skipped")
                logger.warning({str(e)})
                fallback = None
            self.g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback, unk='')
        elif lang_code == 'j':
            try:
                from misaki import ja
                self.g2p = ja.JAG2P()
            except ImportError:
                logger.error("You need to `pip install misaki[ja]` to use lang_code='j'")
                raise
        elif lang_code == 'z':
            try:
                from misaki import zh
                self.g2p = zh.ZHG2P()
            except ImportError:
                logger.error("You need to `pip install misaki[zh]` to use lang_code='z'")
                raise
        else:
            language = LANG_CODES[lang_code]
            logger.warning(f"Using EspeakG2P(language='{language}').  Streaming may not work as efficiently as English.") #Note added regarding efficiency of other languages than 'ab'
            self.g2p = espeak.EspeakG2P(language=language)

    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
        else:
            f = hf_hub_download(repo_id=KModel.REPO_ID, filename=f'voices/{voice}.pt')
            if not voice.startswith(self.lang_code):
                v = LANG_CODES.get(voice, voice)
                p = LANG_CODES.get(self.lang_code, self.lang_code)
                logger.warning(f'Language mismatch, loading {v} voice into {p} pipeline.')
        pack = torch.load(f, weights_only=True)
        self.voices[voice] = pack
        return pack

    def load_voice(self, voice: str, delimiter: str = ",") -> torch.FloatTensor:
        if voice in self.voices:
            return self.voices[voice]
        logger.debug(f"Loading voice: {voice}")
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            return packs[0]
        self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
        return self.voices[voice]


    def _process_phoneme(self, phoneme: str, pack: torch.FloatTensor, speed: float, prev_audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Processes a single phoneme and generates audio autoregressively."""
        if not phoneme:
            return torch.tensor([])

        try:
           
            input_ids = list(filter(lambda i: i is not None, map(lambda p: self.model.vocab.get(p), phoneme)))
            input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.model.device)
            input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(self.model.device)
            text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
            text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.model.device)

            bert_dur = self.model.bert(input_ids, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            ref_s = pack.to(self.model.device)
            s = ref_s[:, 128:]
            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

            # Autoregressive part:  Predict one phoneme at a time.
            indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.model.device), pred_dur)
            pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.model.device)
            pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
            pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.model.device)

            en = d.transpose(-1, -2) @ pred_aln_trg
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)  # Get F0 and N

            t_en = self.model.text_encoder(input_ids, input_lengths, text_mask) # Convert phonemes to input IDs
            asr = t_en @ pred_aln_trg
            audio = self.model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu()

            return audio

        except Exception as e:
            logger.error(f"Error processing phoneme: {e}")
            return torch.tensor([])


    def __call__(
        self,
        text: Union[str, List[str]],
        voice: Optional[str] = None,
        speed: Number = 1,
        split_pattern: Optional[str] = r'\n+'
    ) -> Generator[torch.FloatTensor, None, None]:

        if self.model is None:
            raise ValueError("A model is required for audio generation.")
        if voice is None:
            raise ValueError('Specify a voice.')

        pack = self.load_voice(voice).to(self.model.device)
        prev_audio = None

        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]

        for sentence in text:
            if self.lang_code in 'ab':
                _, tokens = self.g2p(sentence)
                phonemes = ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()
            else:
                phonemes = self.g2p(sentence)
            
            if not phonemes: continue # Skip if phonemes are empty after g2p
            
            phoneme_list = list(phonemes)  # Convert to list for individual char access
            
            for phoneme in phoneme_list:
                audio_chunk = self._process_phoneme(phoneme, pack, speed, prev_audio)
                if audio_chunk.numel() > 0: # Check if audio_chunk is not empty
                    yield audio_chunk
                #  prev_audio = audio_chunk #Removed this line which will lead to generate phonemes in one pass instead of multiple
