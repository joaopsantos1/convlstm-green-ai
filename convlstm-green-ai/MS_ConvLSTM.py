from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

RecurrentState = tuple[torch.Tensor, torch.Tensor]


class MSConvLSTMCell(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        cnn_dropout: float,
        rnn_dropout: float,
        peephole: bool,
        layer_norm: bool,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.hidden_channels = hidden_channels
        self.peephole = peephole
        self.input_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.hidden_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.input_dropout = nn.Dropout2d(cnn_dropout)
        self.hidden_dropout = nn.Dropout2d(rnn_dropout)
        self.gate_norm = (
            nn.GroupNorm(1, 4 * hidden_channels) if layer_norm else nn.Identity()
        )
        self.cell_norm = (
            nn.GroupNorm(1, hidden_channels) if layer_norm else nn.Identity()
        )

        if peephole:
            self.weight_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            self.weight_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            self.weight_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

    def forward(self, x: torch.Tensor, state: RecurrentState) -> RecurrentState:
        hidden, cell = state
        gates = self.gate_norm(
            self.input_conv(self.input_dropout(x))
            + self.hidden_conv(self.hidden_dropout(hidden))
        )
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)

        if self.peephole:
            input_gate = input_gate + self.weight_ci * cell
            forget_gate = forget_gate + self.weight_cf * cell

        next_cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(
            input_gate
        ) * torch.tanh(candidate)
        next_cell = self.cell_norm(next_cell)

        if self.peephole:
            output_gate = output_gate + self.weight_co * next_cell

        next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
        return next_hidden, next_cell


class MS_ConvLSTM(nn.Module):
    """Multi-scale ConvLSTM encoder-forecaster with full/half/quarter scales."""

    def __init__(
        self,
        in_channels: int = 1,
        output_seq_len: int = 6,
        hidden_dims: Sequence[int] = (64, 64),
        kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        rnn_dropout: float = 0.2,
        peephole: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive channel counts.")
        if output_seq_len <= 0:
            raise ValueError("output_seq_len must be positive.")

        self.in_channels = in_channels
        self.output_seq_len = output_seq_len
        self.hidden_channels = hidden_dims[-1]
        self.embed = nn.Conv2d(
            in_channels=in_channels, out_channels=self.hidden_channels, kernel_size=1
        )
        self.encoder_cells = self._make_cells(
            self.hidden_channels,
            kernel_size,
            cnn_dropout,
            rnn_dropout,
            peephole,
            layer_norm,
        )
        self.forecaster_cells = self._make_cells(
            self.hidden_channels,
            kernel_size,
            cnn_dropout,
            rnn_dropout,
            peephole,
            layer_norm,
        )
        self.downs = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(2)]
        )
        self.readout = nn.Conv2d(
            in_channels=self.hidden_channels, out_channels=in_channels, kernel_size=1
        )

    @staticmethod
    def _make_cells(
        hidden_channels: int,
        kernel_size: int,
        cnn_dropout: float,
        rnn_dropout: float,
        peephole: bool,
        layer_norm: bool,
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                MSConvLSTMCell(
                    hidden_channels,
                    kernel_size,
                    cnn_dropout,
                    rnn_dropout,
                    peephole,
                    layer_norm,
                )
                for _ in range(6)
            ]
        )

    def _initial_states(self, frame: torch.Tensor) -> list[RecurrentState]:
        batch_size, _, height, width = frame.shape
        half_height, half_width = height // 2, width // 2
        quarter_height, quarter_width = half_height // 2, half_width // 2
        spatial_sizes = (
            (height, width),
            (half_height, half_width),
            (quarter_height, quarter_width),
            (quarter_height, quarter_width),
            (half_height, half_width),
            (height, width),
        )
        return [
            (
                frame.new_zeros(
                    (batch_size, self.hidden_channels, state_height, state_width)
                ),
                frame.new_zeros(
                    (batch_size, self.hidden_channels, state_height, state_width)
                ),
            )
            for state_height, state_width in spatial_sizes
        ]

    def _run_multiscale(
        self,
        frame: torch.Tensor,
        cells: nn.ModuleList,
        states: list[RecurrentState],
    ) -> tuple[torch.Tensor, list[RecurrentState]]:
        frame = self.embed(frame)
        stage_outputs = []
        next_states = []

        for index, cell in enumerate(cells):
            hidden, cell_state = cell(frame, states[index])
            next_states.append((hidden, cell_state))
            stage_outputs.append(hidden)
            frame = hidden

            if index in (0, 1):
                frame = self.downs[index](frame)
            elif index in (3, 4):
                skip = stage_outputs[1] if index == 3 else stage_outputs[0]
                frame = F.interpolate(
                    frame,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                frame = frame + skip

        return frame, next_states

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != self.in_channels or x.shape[1] == 0:
            raise ValueError(
                "x must have shape (batch, time, channels, height, width) "
                "with at least one frame."
            )
        if target is not None and (
            target.ndim != 5
            or target.shape[0] != x.shape[0]
            or target.shape[2:] != x.shape[2:]
        ):
            raise ValueError("target must match x in batch, channels, height, and width.")
        if min(x.shape[-2:]) < 4:
            raise ValueError("MS_ConvLSTM requires height and width of at least 4.")
        if target is not None and target.shape[1] < self.output_seq_len:
            raise ValueError("target does not have enough steps for teacher forcing.")

        encoder_states = self._initial_states(x[:, 0])
        for frame in x.unbind(dim=1):
            _, encoder_states = self._run_multiscale(
                frame, self.encoder_cells, encoder_states
            )

        forecasts = []
        frame = x[:, -1]
        states = encoder_states
        for step in range(self.output_seq_len):
            hidden, states = self._run_multiscale(
                frame, self.forecaster_cells, states
            )
            prediction = self.readout(hidden)
            forecasts.append(prediction)

            use_target = (
                target is not None
                and self.training
                and torch.rand((), device=x.device) < teacher_forcing_ratio
            )
            frame = target[:, step] if use_target else prediction

        return torch.stack(forecasts, dim=1)
