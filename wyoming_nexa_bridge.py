#!/usr/bin/env python3
"""
Wyoming Protocol Bridge → Nexa ASR (Parakeet NPU)

This bridge allows Home Assistant to use the Nexa SDK's ASR model
(running on an Android tablet with NPU) as if it were a local
Wyoming-compatible STT service (like faster-whisper).

Architecture:
  Home Assistant ──(Wyoming Protocol)──► This Bridge ──(REST API)──► Tablet NPU
       ◄── transcript text ◄────────────────────────────── { "text": "..." }

Usage:
  python wyoming_nexa_bridge.py --nexa-url http://TABLET_IP:8080 --language es
"""

import argparse
import asyncio
import io
import logging
import wave
from functools import partial

import aiohttp
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)


class NexaAsrHandler(AsyncEventHandler):
    """Handles Wyoming STT events and bridges them to Nexa REST API."""

    def __init__(
        self,
        *args,
        nexa_url: str,
        language: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nexa_url = nexa_url
        self.language = language
        self.audio_data = bytearray()
        self.sample_rate = 16000
        self.sample_width = 2  # 16-bit
        self.channels = 1      # mono

    async def handle_event(self, event: Event) -> bool:
        """Process Wyoming protocol events."""

        if Describe.is_type(event.type):
            # Home Assistant asks "what are you?"
            info = Info(
                asr=[
                    AsrProgram(
                        name="nexa-parakeet-npu",
                        description="Nexa Parakeet ASR on Android NPU",
                        version="1.0.0",
                        attribution=Attribution(
                            name="Nexa AI + NVIDIA Parakeet",
                            url="https://nexa.ai",
                        ),
                        installed=True,
                        models=[
                            AsrModel(
                                name="parakeet-tdt-0.6b-v3",
                                description="Parakeet TDT 0.6B v3 (NPU)",
                                version="0.6.3",
                                attribution=Attribution(
                                    name="NVIDIA",
                                    url="https://nvidia.com",
                                ),
                                installed=True,
                                languages=[self.language],
                            )
                        ],
                    )
                ],
            )
            await self.write_event(info.event())
            _LOGGER.debug("Sent info to Home Assistant")
            return True

        if AudioStart.is_type(event.type):
            # Audio stream starting — prepare buffer
            audio_start = AudioStart.from_event(event)
            self.sample_rate = audio_start.rate
            self.sample_width = audio_start.width
            self.channels = audio_start.channels
            self.audio_data = bytearray()
            _LOGGER.info(
                "Audio format from Wyoming: rate=%d Hz, width=%d bytes, channels=%d",
                self.sample_rate,
                self.sample_width,
                self.channels,
            )
            return True

        if AudioChunk.is_type(event.type):
            # Receiving audio data — append to buffer
            chunk = AudioChunk.from_event(event)
            self.audio_data.extend(chunk.audio)
            return True

        if Transcribe.is_type(event.type):
            # HA sends language info, but Parakeet auto-detects language
            # and crashes with "es" code. Always use "en" instead.
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                _LOGGER.info("HA requested language '%s', but using 'en' (Parakeet auto-detects)", transcribe.language)
            return True

        if AudioStop.is_type(event.type):
            # Audio stream ended — process the audio!
            _LOGGER.info(
                "Audio received: %.1f seconds (%d bytes)",
                len(self.audio_data) / (self.sample_rate * self.sample_width * self.channels),
                len(self.audio_data),
            )

            # Convert to 16kHz mono 16-bit PCM WAV (required by Parakeet)
            audio_bytes = bytes(self.audio_data)

            # Resample if needed
            target_rate = 16000
            target_channels = 1
            target_width = 2  # 16-bit

            if self.channels > 1:
                # Convert stereo to mono by averaging channels
                import struct
                samples = struct.unpack(f"<{len(audio_bytes) // self.sample_width}h", audio_bytes)
                mono_samples = []
                for i in range(0, len(samples), self.channels):
                    avg = sum(samples[i:i + self.channels]) // self.channels
                    mono_samples.append(avg)
                audio_bytes = struct.pack(f"<{len(mono_samples)}h", *mono_samples)
                _LOGGER.info("Converted %d channels to mono", self.channels)

            if self.sample_rate != target_rate:
                # Simple linear resampling
                import struct
                samples = struct.unpack(f"<{len(audio_bytes) // target_width}h", audio_bytes)
                ratio = target_rate / self.sample_rate
                new_length = int(len(samples) * ratio)
                resampled = []
                for i in range(new_length):
                    src_idx = i / ratio
                    idx = int(src_idx)
                    if idx >= len(samples) - 1:
                        resampled.append(samples[-1])
                    else:
                        frac = src_idx - idx
                        val = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
                        resampled.append(max(-32768, min(32767, val)))
                audio_bytes = struct.pack(f"<{len(resampled)}h", *resampled)
                _LOGGER.info("Resampled from %d Hz to %d Hz", self.sample_rate, target_rate)

            # Build WAV file in memory (always 16kHz, 16-bit, mono)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(target_channels)
                wav_file.setsampwidth(target_width)
                wav_file.setframerate(target_rate)
                wav_file.writeframes(audio_bytes)
            wav_buffer.seek(0)
            _LOGGER.info("WAV file created: 16kHz, 16-bit, mono, %d bytes", wav_buffer.getbuffer().nbytes)

            # Send to Nexa tablet for transcription
            text = await self._transcribe(wav_buffer)

            _LOGGER.info("Transcript: '%s'", text)

            # Send transcript back to Home Assistant
            await self.write_event(Transcript(text=text).event())

            # Clear buffer
            self.audio_data = bytearray()

            return False  # Close connection after transcript

        return True

    async def _transcribe(self, wav_buffer: io.BytesIO) -> str:
        """Send audio to Nexa tablet API and get transcription."""
        url = f"{self.nexa_url}/v1/audio/transcriptions"

        form = aiohttp.FormData()
        form.add_field(
            "file",
            wav_buffer,
            filename="audio.wav",
            content_type="audio/wav",
        )
        form.add_field("language", self.language)

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=form) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("text", "")
                    else:
                        error_text = await resp.text()
                        _LOGGER.error(
                            "Nexa API error: HTTP %d - %s", resp.status, error_text
                        )
                        return ""
        except aiohttp.ClientError as e:
            _LOGGER.error("Connection to Nexa tablet failed: %s", e)
            return ""
        except Exception as e:
            _LOGGER.error("Unexpected error during transcription: %s", e)
            return ""


async def main():
    parser = argparse.ArgumentParser(
        description="Wyoming Bridge for Nexa ASR (Parakeet NPU)"
    )
    parser.add_argument(
        "--nexa-url",
        required=True,
        help="URL of the Nexa tablet API (e.g. http://192.168.1.93:8080)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Default language for transcription (default: en, auto-detects Spanish)",
    )
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10300",
        help="Wyoming server URI (default: tcp://0.0.0.0:10300)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _LOGGER.info("=" * 60)
    _LOGGER.info("  Wyoming Bridge → Nexa Parakeet NPU")
    _LOGGER.info("  Tablet URL: %s", args.nexa_url)
    _LOGGER.info("  Language:   %s", args.language)
    _LOGGER.info("  Listening:  %s", args.uri)
    _LOGGER.info("=" * 60)

    server = AsyncServer.from_uri(args.uri)

    handler_factory = partial(
        NexaAsrHandler,
        nexa_url=args.nexa_url,
        language=args.language,
    )

    _LOGGER.info("Bridge ready! Waiting for Home Assistant connections...")

    await server.run(handler_factory)


if __name__ == "__main__":
    asyncio.run(main())
