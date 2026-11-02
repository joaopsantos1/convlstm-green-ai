from typing import Optional
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """One ConvLSTM layer with optional peephole connections and GroupNorm.

    The input-to-gates and hidden-to-gates transforms use standard convolutions.

    Peephole weights are per-channel (not per-pixel), and ``GroupNorm(1, C)``
    does not depend on the image height/width.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size=3,
        cnn_dropout=0.0,
        rnn_dropout=0.0,
        peephole=True,
        layer_norm=True,
    ):
        super(ConvLSTMCell, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.hidden_channels = hidden_channels
        self.peephole = peephole

        self.input_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )  # = size, != ch
        self.hidden_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )  # = size, != ch
        self.input_dropout = nn.Dropout2d(cnn_dropout)
        self.hidden_dropout = nn.Dropout2d(rnn_dropout)

        self.gate_norm = (
            nn.GroupNorm(1, 4 * hidden_channels) if layer_norm else nn.Identity()
        )
        self.cell_norm = nn.GroupNorm(1, hidden_channels) if layer_norm else nn.Identity()

        if peephole:
            self.weight_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            self.weight_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            self.weight_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

    def forward(self, x, state):
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

    def initial_state(self, batch_size, height, width, device, dtype):
        shape = (batch_size, self.hidden_channels, height, width)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )


def make_cell_stack(
    in_channels, hidden_dims, kernel_size, cnn_dropout, rnn_dropout, peephole, layer_norm
):
    cells = []
    channels = in_channels
    for hidden_dim in hidden_dims:
        cells.append(
            ConvLSTMCell(
                channels,
                hidden_dim,
                kernel_size,
                cnn_dropout,
                rnn_dropout,
                peephole,
                layer_norm,
            )
        )
        channels = hidden_dim
    return nn.ModuleList(cells)


class ConvLSTMNet(nn.Module):
    """Encoder-forecaster ConvLSTM.

    Input:  (batch, input_seq_len, channels, height, width)
    Output: (batch, output_seq_len, channels, height, width)

    The encoder reads the observed frames; the forecaster starts from the
    encoder states and the last observed frame, and generates one frame per
    step. During training, teacher forcing can feed the real frame back
    instead of the previous prediction.
    """

    def __init__(
        self,
        in_channels=1,
        output_seq_len=6,
        hidden_dims=(64, 64),
        kernel_size=3,
        cnn_dropout=0.2,
        rnn_dropout=0.2,
        peephole=True,
        layer_norm=True,
    ):
        super(ConvLSTMNet, self).__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must have at least one layer.")

        self.in_channels = in_channels
        self.output_seq_len = output_seq_len
        cell_args = (
            in_channels,
            tuple(hidden_dims),
            kernel_size,
            cnn_dropout,
            rnn_dropout,
            peephole,
            layer_norm,
        )
        self.encoder_cells = make_cell_stack(*cell_args)
        self.forecaster_cells = make_cell_stack(*cell_args)
        self.readout = nn.Conv2d(
            in_channels=hidden_dims[-1],
            out_channels=in_channels,
            kernel_size=1,
        )

    @staticmethod
    def _run_stack(cells, frame, states):
        for index, cell in enumerate(cells):
            states[index] = cell(frame, states[index])
            frame = states[index][0]
        return frame, states

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != self.in_channels:
            raise ValueError("x must have shape (batch, time, channels, height, width).")
        if target is not None and target.shape[1] < self.output_seq_len:
            raise ValueError("target does not have enough steps for teacher forcing.")

        batch_size, _, _, height, width = x.shape
        states = [
            cell.initial_state(batch_size, height, width, x.device, x.dtype)
            for cell in self.encoder_cells
        ]
        for frame in x.unbind(dim=1):
            _, states = self._run_stack(self.encoder_cells, frame, states)

        forecasts = []
        frame = x[:, -1]
        for step in range(self.output_seq_len):
            hidden, states = self._run_stack(self.forecaster_cells, frame, states)
            prediction = self.readout(hidden)
            forecasts.append(prediction)

            use_target = (
                target is not None
                and self.training
                and torch.rand((), device=x.device) < teacher_forcing_ratio
            )
            frame = target[:, step] if use_target else prediction

        return torch.stack(forecasts, dim=1)
