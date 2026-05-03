"""RVC (Retrieval-based Voice Conversion) backend for CordBeat.

Converts TTS WAV output through an RVC model to apply a target voice.
Requires the ``rvc`` extra: ``uv sync --extra rvc``.

All torch / transformers imports are lazy so the module can be imported
even when the optional dependencies are not installed.
"""

from __future__ import annotations

import io
import logging
import wave
from math import gcd
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}

# F0 → coarse pitch mapping constants
_F0_MEL_MIN = 1127.0 * 3.912023005428146  # log(1 + 50/700)
_F0_MEL_MAX = 1127.0 * 7.696761286995918  # log(1 + 1100/700)


# ────────────────────────────────────────────────────────────────────────────
# Neural-network class definitions — only materialised when torch is available
# ────────────────────────────────────────────────────────────────────────────

_TORCH_AVAILABLE = False

try:
    import math as _math

    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F  # noqa: N812
    from torch.nn.utils import remove_weight_norm as _remove_wn
    from torch.nn.utils import weight_norm as _wn

    _TORCH_AVAILABLE = True

    _LRELU_SLOPE = 0.1

    def _fused_gate(a: Any, b: Any, n_ch: int) -> Any:
        x = a + b
        return _torch.tanh(x[:, :n_ch, :]) * _torch.sigmoid(x[:, n_ch:, :])

    def _init_weights(m: Any, mean: float = 0.0, std: float = 0.01) -> None:
        if isinstance(m, _nn.Conv1d):
            m.weight.data.normal_(mean, std)

    def _pad_size(kernel_size: int, dilation: int = 1) -> int:
        return (kernel_size * dilation - dilation) // 2

    class WN(_nn.Module):
        def __init__(
            self,
            hidden_channels: int,
            kernel_size: int,
            dilation_rate: int,
            n_layers: int,
            gin_channels: int = 0,
            p_dropout: float = 0,
        ) -> None:
            super().__init__()
            self.hidden_channels = hidden_channels
            self.n_layers = n_layers
            self.in_layers = _nn.ModuleList()
            self.res_skip_layers = _nn.ModuleList()
            self.drop = _nn.Dropout(p_dropout)
            if gin_channels:
                self.cond_layer = _wn(
                    _nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
                )
            for i in range(n_layers):
                d = dilation_rate**i
                self.in_layers.append(
                    _wn(
                        _nn.Conv1d(
                            hidden_channels,
                            2 * hidden_channels,
                            kernel_size,
                            dilation=d,
                            padding=_pad_size(kernel_size, d),
                        )
                    )
                )
                out_ch = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
                self.res_skip_layers.append(_wn(_nn.Conv1d(hidden_channels, out_ch, 1)))

        def forward(self, x: Any, x_mask: Any, g: Any = None) -> Any:
            output = _torch.zeros_like(x)
            if g is not None and hasattr(self, "cond_layer"):
                g = self.cond_layer(g)
            for i in range(self.n_layers):
                x_in = self.in_layers[i](x)
                g_l = (
                    g[
                        :,
                        i * 2 * self.hidden_channels : (i + 1)
                        * 2
                        * self.hidden_channels,  # noqa: E501
                        :,
                    ]
                    if g is not None
                    else _torch.zeros_like(x_in)
                )
                acts = self.drop(_fused_gate(x_in, g_l, self.hidden_channels))
                rs = self.res_skip_layers[i](acts)
                if i < self.n_layers - 1:
                    x = (x + rs[:, : self.hidden_channels, :]) * x_mask
                    output = output + rs[:, self.hidden_channels :, :]
                else:
                    output = output + rs
            return output * x_mask

        def remove_weight_norm(self) -> None:
            if hasattr(self, "cond_layer"):
                _remove_wn(self.cond_layer)
            for layer in self.in_layers:
                _remove_wn(layer)
            for layer in self.res_skip_layers:
                _remove_wn(layer)

    class ResidualCouplingLayer(_nn.Module):
        def __init__(
            self,
            channels: int,
            hidden_channels: int,
            kernel_size: int,
            dilation_rate: int,
            n_layers: int,
            p_dropout: float = 0,
            gin_channels: int = 0,
            mean_only: bool = False,
        ) -> None:
            super().__init__()
            self.half_channels = channels // 2
            self.mean_only = mean_only
            self.pre = _nn.Conv1d(self.half_channels, hidden_channels, 1)
            self.enc = WN(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                p_dropout=p_dropout,
                gin_channels=gin_channels,
            )
            self.post = _nn.Conv1d(
                hidden_channels, self.half_channels * (2 - int(mean_only)), 1
            )
            self.post.weight.data.zero_()
            self.post.bias.data.zero_()  # type: ignore[union-attr]

        def forward(
            self, x: Any, x_mask: Any, g: Any = None, reverse: bool = False
        ) -> Any:
            x0, x1 = _torch.split(x, [self.half_channels] * 2, 1)
            h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
            stats = self.post(h) * x_mask
            if self.mean_only:
                m, logs = stats, _torch.zeros_like(stats)
            else:
                m, logs = _torch.split(stats, [self.half_channels] * 2, 1)
            if reverse:
                x1 = (x1 - m) * _torch.exp(-logs) * x_mask
            else:
                x1 = (m + x1 * _torch.exp(logs)) * x_mask
            return _torch.cat([x0, x1], 1)

        def remove_weight_norm(self) -> None:
            self.enc.remove_weight_norm()

    class Flip(_nn.Module):
        def forward(  # type: ignore[override]
            self, x: Any, *args: Any, reverse: bool = False, **kwargs: Any
        ) -> Any:
            return _torch.flip(x, [1])

    class ResidualCouplingBlock(_nn.Module):
        def __init__(
            self,
            channels: int,
            hidden_channels: int,
            kernel_size: int,
            dilation_rate: int,
            n_layers: int,
            n_flow_layers: int = 4,
            gin_channels: int = 0,
        ) -> None:
            super().__init__()
            self.flows = _nn.ModuleList()
            for _ in range(n_flow_layers):
                self.flows.append(
                    ResidualCouplingLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    )
                )
                self.flows.append(Flip())

        def forward(
            self, x: Any, x_mask: Any, g: Any = None, reverse: bool = False
        ) -> Any:
            itr = reversed(self.flows) if reverse else self.flows
            for flow in itr:
                x = flow(x, x_mask, g=g, reverse=reverse)
            return x

        def remove_weight_norm(self) -> None:
            for f in self.flows:
                if hasattr(f, "remove_weight_norm"):
                    f.remove_weight_norm()

    class LayerNorm(_nn.Module):
        def __init__(self, channels: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.channels = channels
            self.eps = eps
            self.gamma = _nn.Parameter(_torch.ones(channels))
            self.beta = _nn.Parameter(_torch.zeros(channels))

        def forward(self, x: Any) -> Any:
            return _F.layer_norm(
                x.transpose(1, -1),
                (self.channels,),
                self.gamma,
                self.beta,
                self.eps,
            ).transpose(1, -1)

    class MultiHeadAttention(_nn.Module):
        def __init__(
            self,
            channels: int,
            out_channels: int,
            n_heads: int,
            p_dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.k_channels = channels // n_heads
            self.conv_q = _nn.Conv1d(channels, channels, 1)
            self.conv_k = _nn.Conv1d(channels, channels, 1)
            self.conv_v = _nn.Conv1d(channels, channels, 1)
            self.conv_o = _nn.Conv1d(channels, out_channels, 1)
            self.drop = _nn.Dropout(p_dropout)

        def forward(self, x: Any, c: Any, attn_mask: Any = None) -> Any:
            b, d, t = x.shape
            q = self.conv_q(x).view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
            k = (
                self.conv_k(c)
                .view(b, self.n_heads, self.k_channels, -1)
                .transpose(2, 3)
            )
            v = (
                self.conv_v(c)
                .view(b, self.n_heads, self.k_channels, -1)
                .transpose(2, 3)
            )
            scores = _torch.matmul(q, k.transpose(-2, -1)) / _math.sqrt(self.k_channels)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e4)
            output = _torch.matmul(self.drop(_F.softmax(scores, dim=-1)), v)
            return self.conv_o(output.transpose(2, 3).contiguous().view(b, d, t))

    class FFN(_nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            filter_channels: int,
            kernel_size: int,
            p_dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.conv_1 = _nn.Conv1d(
                in_channels,
                filter_channels,
                kernel_size,
                padding=kernel_size // 2,
            )
            self.conv_2 = _nn.Conv1d(
                filter_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            )
            self.drop = _nn.Dropout(p_dropout)

        def forward(self, x: Any, x_mask: Any) -> Any:
            x = self.drop(_torch.relu(self.conv_1(x * x_mask)))
            return self.conv_2(x * x_mask) * x_mask

    class AttentionEncoder(_nn.Module):
        def __init__(
            self,
            hidden_channels: int,
            filter_channels: int,
            n_heads: int,
            n_layers: int,
            kernel_size: int = 1,
            p_dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.attn_layers = _nn.ModuleList()
            self.norm_layers_1 = _nn.ModuleList()
            self.ffn_layers = _nn.ModuleList()
            self.norm_layers_2 = _nn.ModuleList()
            for _ in range(n_layers):
                self.attn_layers.append(
                    MultiHeadAttention(
                        hidden_channels, hidden_channels, n_heads, p_dropout
                    )
                )
                self.norm_layers_1.append(LayerNorm(hidden_channels))
                self.ffn_layers.append(
                    FFN(
                        hidden_channels,
                        hidden_channels,
                        filter_channels,
                        kernel_size,
                        p_dropout,
                    )
                )
                self.norm_layers_2.append(LayerNorm(hidden_channels))

        def forward(self, x: Any, x_mask: Any) -> Any:
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
            x = x * x_mask
            for attn, n1, ffn, n2 in zip(
                self.attn_layers,
                self.norm_layers_1,
                self.ffn_layers,
                self.norm_layers_2,
            ):
                y = n1(x + attn(x, x, attn_mask))
                x = n2(y + ffn(y, x_mask))
            return x * x_mask

    class TextEncoder768(_nn.Module):
        """RVC v2 content encoder (768-dim HuBERT features)."""

        def __init__(
            self,
            out_channels: int,
            hidden_channels: int,
            filter_channels: int,
            n_heads: int,
            n_layers: int,
            kernel_size: int,
            p_dropout: float,
            f0: bool = True,
        ) -> None:
            super().__init__()
            self.out_channels = out_channels
            self.emb_phone = _nn.Linear(768, hidden_channels)
            if f0:
                self.emb_pitch = _nn.Embedding(256, hidden_channels)
            self.encoder = AttentionEncoder(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
            )
            self.proj = _nn.Conv1d(hidden_channels, out_channels * 2, 1)

        def forward(self, phone: Any, pitch: Any, lengths: Any) -> Any:
            x = self.emb_phone(phone)
            if pitch is not None and hasattr(self, "emb_pitch"):
                x = x + self.emb_pitch(pitch)
            x = x.transpose(1, 2)
            x_mask = (
                (
                    _torch.arange(x.size(2), device=x.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                .unsqueeze(1)
                .to(x.dtype)
            )
            x = self.encoder(x, x_mask)
            m, logs = _torch.split(self.proj(x) * x_mask, self.out_channels, dim=1)
            return m, logs, x_mask

    class TextEncoder256(_nn.Module):
        """RVC v1 content encoder (256-dim HuBERT features)."""

        def __init__(
            self,
            out_channels: int,
            hidden_channels: int,
            filter_channels: int,
            n_heads: int,
            n_layers: int,
            kernel_size: int,
            p_dropout: float,
            f0: bool = True,
        ) -> None:
            super().__init__()
            self.out_channels = out_channels
            self.emb_phone = _nn.Linear(256, hidden_channels)
            if f0:
                self.emb_pitch = _nn.Embedding(256, hidden_channels)
            self.encoder = AttentionEncoder(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
            )
            self.proj = _nn.Conv1d(hidden_channels, out_channels * 2, 1)

        def forward(self, phone: Any, pitch: Any, lengths: Any) -> Any:
            x = self.emb_phone(phone)
            if pitch is not None and hasattr(self, "emb_pitch"):
                x = x + self.emb_pitch(pitch)
            x = x.transpose(1, 2)
            x_mask = (
                (
                    _torch.arange(x.size(2), device=x.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                .unsqueeze(1)
                .to(x.dtype)
            )
            x = self.encoder(x, x_mask)
            m, logs = _torch.split(self.proj(x) * x_mask, self.out_channels, dim=1)
            return m, logs, x_mask

    class SineGen(_nn.Module):
        def __init__(
            self,
            samp_rate: int,
            harmonic_num: int = 0,
            sine_amp: float = 0.1,
            noise_std: float = 0.003,
            voiced_threshold: float = 0,
        ) -> None:
            super().__init__()
            self.sampling_rate = samp_rate
            self.harmonic_num = harmonic_num
            self.sine_amp = sine_amp
            self.noise_std = noise_std
            self.voiced_threshold = voiced_threshold

        def forward(self, f0: Any) -> Any:
            with _torch.no_grad():
                voiced = (f0 > self.voiced_threshold).float()
                mul = _torch.arange(
                    1,
                    self.harmonic_num + 2,
                    device=f0.device,
                    dtype=f0.dtype,
                ).view(1, 1, -1)
                f0h = f0 * mul
                phase = _torch.cumsum(f0h / self.sampling_rate, dim=1) % 1.0
                sines = _torch.sin(2 * _math.pi * phase) * self.sine_amp * voiced
                noise = _torch.randn_like(sines[:, :, :1]) * self.noise_std
            return sines, voiced, noise

    class SourceModuleHnNSF(_nn.Module):
        def __init__(
            self,
            sample_rate: int,
            harmonic_num: int = 0,
            sine_amp: float = 0.1,
            add_noise_std: float = 0.003,
            voiced_threshold: float = 0,
        ) -> None:
            super().__init__()
            self.l_sin_gen = SineGen(
                sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
            )
            self.l_linear = _nn.Linear(harmonic_num + 1, 1)
            self.l_tanh = _nn.Tanh()

        def forward(self, x: Any) -> Any:
            sines, voiced, noise = self.l_sin_gen(x)
            sines = sines.to(self.l_linear.weight.dtype)
            return self.l_tanh(self.l_linear(sines))

    class ResBlock1(_nn.Module):
        def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            dilation: tuple[int, ...] = (1, 3, 5),
        ) -> None:
            super().__init__()
            self.convs1 = _nn.ModuleList(
                [
                    _wn(
                        _nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=d,
                            padding=_pad_size(kernel_size, d),
                        )
                    )
                    for d in dilation
                ]
            )
            self.convs1.apply(_init_weights)
            self.convs2 = _nn.ModuleList(
                [
                    _wn(
                        _nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            padding=_pad_size(kernel_size, 1),
                        )
                    )
                    for _ in dilation
                ]
            )
            self.convs2.apply(_init_weights)

        def forward(self, x: Any, x_mask: Any = None) -> Any:
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = c1(_F.leaky_relu(x, _LRELU_SLOPE))
                xt = c2(_F.leaky_relu(xt, _LRELU_SLOPE))
                x = xt + x
            if x_mask is not None:
                x = x * x_mask
            return x

        def remove_weight_norm(self) -> None:
            for layer in self.convs1:
                _remove_wn(layer)
            for layer in self.convs2:
                _remove_wn(layer)

    class ResBlock2(_nn.Module):
        def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            dilation: tuple[int, ...] = (1, 3),
        ) -> None:
            super().__init__()
            self.convs = _nn.ModuleList(
                [
                    _wn(
                        _nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=d,
                            padding=_pad_size(kernel_size, d),
                        )
                    )
                    for d in dilation
                ]
            )
            self.convs.apply(_init_weights)

        def forward(self, x: Any, x_mask: Any = None) -> Any:
            for c in self.convs:
                xt = c(_F.leaky_relu(x, _LRELU_SLOPE))
                x = xt + x
            if x_mask is not None:
                x = x * x_mask
            return x

        def remove_weight_norm(self) -> None:
            for layer in self.convs:
                _remove_wn(layer)

    class GeneratorNSF(_nn.Module):
        def __init__(
            self,
            initial_channel: int,
            resblock: str,
            resblock_kernel_sizes: list[int],
            resblock_dilation_sizes: list[list[int]],
            upsample_rates: list[int],
            upsample_initial_channel: int,
            upsample_kernel_sizes: list[int],
            gin_channels: int,
            sr: int,
            is_half: bool = False,
        ) -> None:
            super().__init__()
            self.num_kernels = len(resblock_kernel_sizes)
            self.num_upsamples = len(upsample_rates)
            self.upp = _math.prod(upsample_rates)
            self.f0_upsamp = _nn.Upsample(scale_factor=self.upp)
            self.m_source = SourceModuleHnNSF(sample_rate=sr)
            self.is_half = is_half
            self.conv_pre = _nn.Conv1d(
                initial_channel, upsample_initial_channel, 7, 1, padding=3
            )
            resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
            self.ups = _nn.ModuleList()
            self.noise_convs = _nn.ModuleList()
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                c_cur = upsample_initial_channel // (2 ** (i + 1))
                self.ups.append(
                    _wn(
                        _nn.ConvTranspose1d(
                            upsample_initial_channel // (2**i),
                            c_cur,
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    )
                )
                stride_f0 = (
                    _math.prod(upsample_rates[i + 1 :])
                    if i + 1 < len(upsample_rates)
                    else 1
                )
                self.noise_convs.append(
                    _nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2 if stride_f0 > 1 else 1,
                        stride=stride_f0,
                        padding=stride_f0 // 2 if stride_f0 > 1 else 0,
                    )
                )
            self.resblocks = _nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                    self.resblocks.append(resblock_cls(ch, k, tuple(d)))
            self.conv_post = _nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
            self.ups.apply(_init_weights)
            if gin_channels:
                self.cond = _nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        def forward(self, x: Any, f0: Any, g: Any = None) -> Any:
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            if self.is_half:
                f0 = f0.float()
            har_source = self.m_source(f0)
            if self.is_half:
                har_source = har_source.half()
            har_source = har_source.transpose(1, 2)
            x = self.conv_pre(x)
            if g is not None and hasattr(self, "cond"):
                x = x + self.cond(g)
            for i in range(self.num_upsamples):
                x = _F.leaky_relu(x, _LRELU_SLOPE)
                x = self.ups[i](x)
                x_source = self.noise_convs[i](har_source)
                x = x + x_source[:, :, : x.shape[2]]
                xs = None
                for j in range(self.num_kernels):
                    r = self.resblocks[i * self.num_kernels + j](x)
                    xs = r if xs is None else xs + r
                x = xs / self.num_kernels  # type: ignore[operator]
            x = _F.leaky_relu(x)
            x = self.conv_post(x)
            return _torch.tanh(x)

        def remove_weight_norm(self) -> None:
            for layer in self.ups:
                _remove_wn(layer)
            for layer in self.resblocks:
                layer.remove_weight_norm()

    class PosteriorEncoder(_nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            kernel_size: int,
            dilation_rate: int,
            n_layers: int,
            gin_channels: int = 0,
        ) -> None:
            super().__init__()
            self.out_channels = out_channels
            self.pre = _nn.Conv1d(in_channels, hidden_channels, 1)
            self.enc = WN(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels=gin_channels,
            )
            self.proj = _nn.Conv1d(hidden_channels, out_channels * 2, 1)

        def forward(self, x: Any, x_lengths: Any, g: Any = None) -> None:
            pass  # Not used during inference

    class SynthesizerTrnMs768NSFSid(_nn.Module):
        """RVC v2 synthesizer (768-dim content features, with F0)."""

        def __init__(
            self,
            spec_channels: int,
            segment_size: int,
            inter_channels: int,
            hidden_channels: int,
            filter_channels: int,
            n_heads: int,
            n_layers: int,
            kernel_size: int,
            p_dropout: float,
            resblock: str,
            resblock_kernel_sizes: list[int],
            resblock_dilation_sizes: list[list[int]],
            upsample_rates: list[int],
            upsample_initial_channel: int,
            upsample_kernel_sizes: list[int],
            spk_embed_dim: int,
            gin_channels: int,
            sr: int,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            is_half = kwargs.get("is_half", False)
            self.enc_p = TextEncoder768(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
            )
            self.dec = GeneratorNSF(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels,
                sr,
                is_half=is_half,
            )
            self.enc_q = PosteriorEncoder(
                spec_channels,
                inter_channels,
                hidden_channels,
                5,
                1,
                16,
                gin_channels=gin_channels,
            )
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                3,
                gin_channels=gin_channels,
            )
            self.emb_g = _nn.Embedding(spk_embed_dim, gin_channels)

        @_torch.no_grad()
        def infer(
            self, phone: Any, phone_lengths: Any, pitch: Any, pitchf: Any, ds: Any
        ) -> Any:
            g = self.emb_g(ds).unsqueeze(-1)
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + _torch.randn_like(m_p) * _torch.exp(logs_p)) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            return self.dec(z * x_mask, pitchf, g=g), x_mask

    class SynthesizerTrnMs256NSFSid(_nn.Module):
        """RVC v1 synthesizer (256-dim content features, with F0)."""

        def __init__(
            self,
            spec_channels: int,
            segment_size: int,
            inter_channels: int,
            hidden_channels: int,
            filter_channels: int,
            n_heads: int,
            n_layers: int,
            kernel_size: int,
            p_dropout: float,
            resblock: str,
            resblock_kernel_sizes: list[int],
            resblock_dilation_sizes: list[list[int]],
            upsample_rates: list[int],
            upsample_initial_channel: int,
            upsample_kernel_sizes: list[int],
            spk_embed_dim: int,
            gin_channels: int,
            sr: int,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            is_half = kwargs.get("is_half", False)
            self.enc_p = TextEncoder256(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
            )
            self.dec = GeneratorNSF(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels,
                sr,
                is_half=is_half,
            )
            self.enc_q = PosteriorEncoder(
                spec_channels,
                inter_channels,
                hidden_channels,
                5,
                1,
                16,
                gin_channels=gin_channels,
            )
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                3,
                gin_channels=gin_channels,
            )
            self.emb_g = _nn.Embedding(spk_embed_dim, gin_channels)

        @_torch.no_grad()
        def infer(
            self, phone: Any, phone_lengths: Any, pitch: Any, pitchf: Any, ds: Any
        ) -> Any:
            g = self.emb_g(ds).unsqueeze(-1)
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + _torch.randn_like(m_p) * _torch.exp(logs_p)) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            return self.dec(z * x_mask, pitchf, g=g), x_mask

except ImportError:
    _TORCH_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# F0 extraction helpers (numpy-dependent, also lazy)
# ────────────────────────────────────────────────────────────────────────────


def _resample(audio: Any, src_sr: int, dst_sr: int) -> Any:
    """High-quality resampling via scipy."""
    if src_sr == dst_sr:
        return audio
    from scipy.signal import resample_poly  # type: ignore[import-not-found]

    g = gcd(src_sr, dst_sr)
    import numpy as _np

    return resample_poly(audio, dst_sr // g, src_sr // g).astype(_np.float32)


def _extract_f0_harvest(audio_16k: Any) -> Any:
    """F0 extraction via pyworld harvest (falls back to YIN)."""
    try:
        import numpy as _np
        import pyworld as pw  # type: ignore[import-not-found]

        f0, t = pw.harvest(
            audio_16k.astype(_np.float64),
            16000,
            f0_floor=50.0,
            f0_ceil=1100.0,
            frame_period=10.0,
        )
        f0 = pw.stonemask(audio_16k.astype(_np.float64), f0, t, 16000)
        return f0.astype(_np.float32)
    except Exception:
        logger.warning("pyworld unavailable — falling back to YIN F0 extraction")
        return _extract_f0_yin(audio_16k)


def _extract_f0_yin(audio_16k: Any) -> Any:
    """Lightweight autocorrelation-based F0 estimation (no external deps)."""
    import numpy as _np

    sr = 16000
    hop = 160  # 100 Hz frame rate
    lag_min = max(2, int(sr / 1100))
    lag_max = int(sr / 50)
    win = lag_max * 2
    n_frames = max(0, (len(audio_16k) - win) // hop)
    f0 = _np.zeros(n_frames, dtype=_np.float32)

    for i in range(n_frames):
        seg = audio_16k[i * hop : i * hop + win].astype(_np.float64)
        seg -= seg.mean()
        x = seg[:lag_max]
        energy = _np.dot(x, x)
        if energy < 1e-8:
            continue
        acf = _np.array(
            [_np.dot(x, seg[t : t + lag_max]) for t in range(lag_min, lag_max + 1)]
        )
        acf /= energy
        peak_lag = 0
        peak_val = -1.0
        for j in range(len(acf)):
            if acf[j] > peak_val:
                peak_val = acf[j]
                peak_lag = lag_min + j
            if peak_val > 0.3 and j > 0 and acf[j] < acf[j - 1]:
                break
        if peak_val > 0.3 and peak_lag > 0:
            f0[i] = sr / peak_lag

    return f0


def _f0_to_coarse(f0: Any) -> Any:
    """Map F0 (Hz) to coarse pitch indices (0–255)."""
    import numpy as _np

    f0_mel = 1127.0 * _np.log(1 + f0 / 700)
    coarse = _np.zeros_like(f0_mel, dtype=_np.int64)
    voiced = f0_mel > 0
    coarse[voiced] = _np.clip(
        1 + (f0_mel[voiced] - _F0_MEL_MIN) * 254 / (_F0_MEL_MAX - _F0_MEL_MIN),
        1,
        255,
    ).astype(_np.int64)
    return coarse


# ────────────────────────────────────────────────────────────────────────────
# RVCBackend class
# ────────────────────────────────────────────────────────────────────────────


class RVCBackend:
    """Encapsulates an RVC model + HuBERT feature extractor.

    Usage::

        rvc = RVCBackend()
        rvc.load("path/to/model.pth", f0_up_key=0)
        output_wav = rvc.convert(input_wav_bytes)

    ``convert()`` is synchronous and CPU/GPU bound; call it from a thread
    pool executor (e.g. ``asyncio.get_event_loop().run_in_executor``).
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._hubert: Any = None
        self._device: Any = None
        self._target_sr: int = 40000
        self._is_half: bool = False
        self._version: str = "v2"
        self._f0_up_key: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        model_path: str,
        index_path: str | None = None,
        f0_up_key: int = 0,
        device: str | None = None,
    ) -> None:
        """Load the RVC model and HuBERT feature extractor.

        Args:
            model_path:  Path to the ``.pth`` checkpoint file.
            index_path:  Optional ``.index`` file (currently unused, reserved).
            f0_up_key:   Pitch shift in semitones (0 = no shift).
            device:      Compute device (``"cuda"``, ``"cpu"``). Auto-detects if None.
        """
        if not _TORCH_AVAILABLE:
            logger.error("torch is not installed. Install with: uv sync --extra rvc")
            return

        self._f0_up_key = f0_up_key
        model_file = Path(model_path)
        if not model_file.exists():
            logger.error("RVC model not found: %s", model_file)
            return

        # Device selection
        if device:
            self._device = _torch.device(device)  # type: ignore[name-defined]
        else:
            self._device = _torch.device(  # type: ignore[name-defined]
                "cuda" if _torch.cuda.is_available() else "cpu"  # type: ignore[name-defined]
            )
        self._is_half = self._device.type == "cuda"

        # ── 1. RVC checkpoint ──────────────────────────────────────────────
        logger.info("Loading RVC model: %s", model_file)
        try:
            cpt = _torch.load(  # type: ignore[name-defined]
                str(model_file), map_location="cpu", weights_only=False
            )
        except Exception:
            logger.exception("Failed to load RVC checkpoint: %s", model_file)
            return

        model_cfg = cpt.get("config")
        if model_cfg is None:
            logger.error("RVC checkpoint has no 'config' key: %s", model_file)
            return

        self._version = str(cpt.get("info", "v2"))
        if "v1" not in self._version and "v2" not in self._version:
            self._version = "v2" if len(model_cfg) > 17 else "v1"

        sr_raw = cpt.get("sr", "40k")
        self._target_sr = (
            _SR_MAP.get(str(sr_raw), 40000) if isinstance(sr_raw, str) else int(sr_raw)
        )

        logger.info(
            "RVC checkpoint: version=%s sr=%s f0=%s config_len=%d",
            self._version,
            sr_raw,
            cpt.get("f0", "?"),
            len(model_cfg),
        )

        model_cls = (
            SynthesizerTrnMs768NSFSid  # type: ignore[name-defined]
            if "v2" in self._version
            else SynthesizerTrnMs256NSFSid  # type: ignore[name-defined]
        )
        try:
            net = model_cls(*model_cfg, is_half=self._is_half)
            net.load_state_dict(cpt["weight"], strict=False)
            net = net.to(self._device).eval()
            if self._is_half:
                net = net.half()
            net.dec.remove_weight_norm()
            self._model = net
        except Exception:
            logger.exception("Failed to build RVC model")
            return

        # ── 2. HuBERT ──────────────────────────────────────────────────────
        hubert_name = "facebook/hubert-base-ls960"
        logger.info("Loading HuBERT: %s (may download on first use)", hubert_name)
        try:
            from transformers import HubertModel  # type: ignore[import-not-found]
        except ImportError:
            logger.error(
                "transformers is not installed. Install with: uv sync --extra rvc"
            )
            self._model = None
            return
        try:
            self._hubert = (
                HubertModel.from_pretrained(hubert_name).to(self._device).eval()
            )
            if self._is_half:
                self._hubert = self._hubert.half()
        except Exception:
            logger.exception("Failed to load HuBERT")
            self._model = None
            return

        logger.info(
            "RVC ready (version=%s sr=%d half=%s f0_up_key=%d)",
            self._version,
            self._target_sr,
            self._is_half,
            self._f0_up_key,
        )

    def is_loaded(self) -> bool:
        """Return True if the model has been loaded successfully."""
        return self._model is not None

    def unload(self) -> None:
        """Release model weights from memory."""
        self._model = None
        self._hubert = None
        logger.info("RVC model unloaded")

    def convert(self, wav_bytes: bytes) -> bytes:
        """Convert *wav_bytes* (WAV) through the RVC model and return WAV bytes.

        This is a synchronous, CPU/GPU-bound operation.
        Returns the original audio unchanged if the model is not loaded or
        if an error occurs.
        """
        return self._convert_sync(wav_bytes)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _convert_sync(self, wav_data: bytes) -> bytes:
        if self._model is None or self._hubert is None:
            return wav_data

        try:
            import time as _time

            import numpy as _np

            t0 = _time.perf_counter()

            # WAV → numpy float32
            buf = io.BytesIO(wav_data)
            with wave.open(buf, "rb") as wf:
                src_sr = wf.getframerate()
                n_ch = wf.getnchannels()
                raw = wf.readframes(wf.getnframes())

            audio = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
            if n_ch > 1:
                audio = audio.reshape(-1, n_ch).mean(axis=1)

            # Resample to 16 kHz
            audio_16k = _resample(audio, src_sr, 16000)
            t_frames = len(audio_16k) // 160
            if t_frames < 1:
                return wav_data
            t1 = _time.perf_counter()

            # HuBERT features
            feats_in = (
                _torch.from_numpy(audio_16k)  # type: ignore[name-defined]
                .float()
                .unsqueeze(0)
                .to(self._device)
            )
            if self._is_half:
                feats_in = feats_in.half()
            with _torch.no_grad():  # type: ignore[name-defined]
                feats = self._hubert(feats_in).last_hidden_state

            import torch.nn.functional as _tf

            feats = _tf.interpolate(
                feats.transpose(1, 2), size=t_frames, mode="nearest"
            ).transpose(1, 2)
            t2 = _time.perf_counter()

            # F0 extraction
            f0 = _extract_f0_harvest(audio_16k)
            f0 = f0 * (2 ** (self._f0_up_key / 12))
            if len(f0) > t_frames:
                f0 = f0[:t_frames]
            elif len(f0) < t_frames:
                f0 = _np.pad(f0, (0, t_frames - len(f0)))
            t3 = _time.perf_counter()

            pitch_coarse = _f0_to_coarse(f0)

            phone = feats.to(self._device)
            phone_lengths = _torch.tensor(  # type: ignore[name-defined]
                [t_frames],
                dtype=_torch.long,
                device=self._device,  # type: ignore[name-defined]
            )
            pitch = (
                _torch.from_numpy(pitch_coarse)  # type: ignore[name-defined]
                .long()
                .unsqueeze(0)
                .to(self._device)
            )
            pitchf = (
                _torch.from_numpy(f0)  # type: ignore[name-defined]
                .float()
                .unsqueeze(0)
                .to(self._device)
            )
            if self._is_half:
                phone = phone.half()
                pitchf = pitchf.half()
            sid = _torch.tensor(  # type: ignore[name-defined]
                [0],
                dtype=_torch.long,
                device=self._device,  # type: ignore[name-defined]
            )

            with _torch.no_grad():  # type: ignore[name-defined]
                audio_out, _ = self._model.infer(
                    phone, phone_lengths, pitch, pitchf, sid
                )
            audio_out = audio_out[0, 0].cpu().float().numpy()
            t4 = _time.perf_counter()

            audio_int16 = _np.clip(audio_out * 32767, -32768, 32767).astype(_np.int16)
            out_buf = io.BytesIO()
            with wave.open(out_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._target_sr)
                wf.writeframes(audio_int16.tobytes())

            ms_resample = int((t1 - t0) * 1000)
            ms_hubert = int((t2 - t1) * 1000)
            ms_f0 = int((t3 - t2) * 1000)
            ms_infer = int((t4 - t3) * 1000)
            logger.debug(
                "RVC profile: resample=%dms hubert=%dms f0=%dms infer=%dms total=%dms",
                ms_resample,
                ms_hubert,
                ms_f0,
                ms_infer,
                ms_resample + ms_hubert + ms_f0 + ms_infer,
            )

            return out_buf.getvalue()

        except Exception:
            logger.exception("RVC conversion error; returning original audio")
            return wav_data
