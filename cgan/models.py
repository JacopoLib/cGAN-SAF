"""Networks of the c-GAN: mixture encoder (attention pooling), generator, critics and latent encoder.

Attribute names are kept stable because trained weights (and the inference bundles) are loaded by name.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

ACTIVATIONS = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, "tanh": nn.Tanh}


def init_weights(module):
    """Xavier-uniform weights and zero biases for every Linear layer."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def maybe_spectral_norm(layer, apply):
    return spectral_norm(layer) if apply else layer


class ResidualBlock(nn.Module):
    """Two linear layers with an activation in between and an identity shortcut."""

    def __init__(self, dim, activation, dropout=0.0, apply_spectral_norm=False):
        super().__init__()
        self.fc1 = maybe_spectral_norm(nn.Linear(dim, dim), apply_spectral_norm)
        self.fc2 = maybe_spectral_norm(nn.Linear(dim, dim), apply_spectral_norm)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.activation(self.fc1(x)))
        return x + self.fc2(out)


class MLP(nn.Module):
    """Feed-forward network with LayerNorm after the input layer; hidden layers are residual blocks or plain layers."""

    def __init__(
        self, inp_dim, out_dim, n_layers, neurons, activation="relu", use_residual=False, apply_spectral_norm=False
    ):
        super().__init__()
        act = ACTIVATIONS.get(activation, nn.Identity)()
        sn = apply_spectral_norm
        self.fc_net = nn.Sequential()
        self.fc_net.add_module("fc_inp", maybe_spectral_norm(nn.Linear(inp_dim, neurons), sn))
        self.fc_net.add_module("layernorm_inp", nn.LayerNorm(neurons))
        self.fc_net.add_module("activation_inp", act)
        for i in range(n_layers):
            if use_residual:
                self.fc_net.add_module(f"residual_block{i + 1}", ResidualBlock(neurons, act, 0.0, sn))
            else:
                self.fc_net.add_module(f"hidden_layer{i + 1}", maybe_spectral_norm(nn.Linear(neurons, neurons), sn))
                self.fc_net.add_module(f"layernorm{i + 1}", nn.LayerNorm(neurons))
                self.fc_net.add_module(f"activation{i + 1}", act)
        self.fc_net.add_module("fc_out", maybe_spectral_norm(nn.Linear(neurons, out_dim), sn))
        self.fc_net.add_module("output_activation", nn.Identity())
        init_weights(self)

    def forward(self, x):
        return self.fc_net(x)


class Critic(nn.Module):
    """Discriminator of one task: score of the pair (x, y)."""

    def __init__(self, x_dim, activation, n_layers, hidden):
        super().__init__()
        self.net = MLP(x_dim + 1, 1, n_layers, hidden, activation, apply_spectral_norm=True)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


class SharedTrunk(nn.Module):
    """Deterministic backbone shared by all tasks: Linear + LayerNorm + act, residual blocks, Linear."""

    def __init__(self, mol_dim, W1, n1, activation="relu", dropout=0.0):
        super().__init__()
        Act = ACTIVATIONS.get(activation, nn.ReLU)
        self.gateway_lin = nn.Linear(mol_dim, W1)
        self.gateway_ln = nn.LayerNorm(W1)
        self.gateway_act = Act()
        self.phase1_blocks = nn.ModuleList([ResidualBlock(W1, Act(), dropout) for _ in range(n1)])
        self.phase1_out = nn.Linear(W1, W1)
        init_weights(self)

    def forward(self, x_mol):
        h = self.gateway_act(self.gateway_ln(self.gateway_lin(x_mol)))
        for block in self.phase1_blocks:
            h = block(h)
        return self.phase1_out(h)


class BridgeNet(nn.Module):
    """Generator head of one task: two-stage residual MLP taking [mixture representation, z] and returning y."""

    def __init__(self, inp_dim, W1, W2, n1, n2, activation="relu", dropout=0.0):
        super().__init__()
        Act = ACTIVATIONS.get(activation, nn.ReLU)
        self.gateway_lin = nn.Linear(inp_dim, W1)
        self.gateway_ln = nn.LayerNorm(W1)
        self.gateway_act = Act()
        self.phase1_blocks = nn.ModuleList([ResidualBlock(W1, Act(), dropout) for _ in range(n1)])
        self.phase1_out = nn.Linear(W1, W1)
        self.bridge_lin = nn.Linear(W1, W2)
        self.bridge_ln = nn.LayerNorm(W2)
        self.bridge_act = Act()
        self.phase2_blocks = nn.ModuleList([ResidualBlock(W2, Act(), dropout) for _ in range(n2)])
        self.output_lin = nn.Linear(W2, 1)
        init_weights(self)

    def forward(self, x):
        h = self.gateway_act(self.gateway_ln(self.gateway_lin(x)))
        for block in self.phase1_blocks:
            h = block(h)
        h = self.bridge_act(self.bridge_ln(self.bridge_lin(self.phase1_out(h))))
        for block in self.phase2_blocks:
            h = block(h)
        return self.output_lin(h)


class Generator(nn.Module):
    """One BridgeNet head per task. With `shared` = (width, depth) the heads sit on a shared SharedTrunk
    (multi-task model); with `shared=None` they take the mixture representation directly (single-task model)."""

    def __init__(self, n_tasks, z_dim, mol_dim, head_widths, head_depths, activation="relu", dropout=0.0, shared=None):
        super().__init__()
        self.n_tasks, self.z_dim, self.mol_dim = n_tasks, z_dim, mol_dim
        W1, W2 = head_widths  # phase-1 / phase-2 width of every head
        n1, n2 = head_depths
        self.shared_trunk = None
        gateway_in = mol_dim
        if shared is not None:
            self.shared_trunk = SharedTrunk(mol_dim, shared[0], shared[1], activation, dropout)
            gateway_in = shared[0]
        self.heads = nn.ModuleList(
            [BridgeNet(gateway_in + z_dim, W1, W2, n1, n2, activation, dropout) for _ in range(n_tasks)]
        )

    def forward(self, x, z):
        """x: (B, mol_dim + reserved columns), z: (B, z_dim) shared by all tasks or (B, n_tasks, z_dim)."""
        h = x[:, : self.mol_dim]
        if self.shared_trunk is not None:
            h = self.shared_trunk(h)
        outs = []
        for t, head in enumerate(self.heads):
            z_t = z if z.dim() == 2 else z[:, t, :]
            outs.append(head(torch.cat([h, z_t], dim=1)))
        return torch.cat(outs, dim=1)


class MultiQueryPooling(nn.Module):
    """Attention pooling with learnable queries and a volume-fraction bias on the scores."""

    def __init__(self, d_model, num_queries):
        super().__init__()
        self.d_model = d_model
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.frac_scale = nn.Parameter(torch.ones(num_queries) * 0.5)
        self.query_w = nn.Parameter(torch.zeros(num_queries))  # softmax weights over queries (uniform at init)

    def forward(self, tokens, mask, frac):
        """tokens: (B, L, d), mask: (B, L) True on padding, frac: (B, L) normalized fractions."""
        B, _, d = tokens.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        scores = torch.matmul(q, tokens.transpose(1, 2)) / (d**0.5)  # (B, K, L)
        scores = scores + self.frac_scale.view(1, -1, 1) * torch.clamp(frac, min=1e-12).log().unsqueeze(1)
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        pooled = torch.matmul(torch.softmax(scores, dim=-1), tokens)  # (B, K, d)
        return (pooled * torch.softmax(self.query_w, dim=0).view(1, -1, 1)).sum(dim=1)


class MixtureEncoder(nn.Module):
    """Mixture representation from the species descriptors of the components.

    Returns the volume-fraction-weighted mean (mean_pool) and the attention correction (blend_delta). For tasks with
    `blend_gate` the correction is multiplied by an is-blend indicator, so it vanishes exactly for pure compounds.
    """

    def __init__(self, latent_dim, num_queries, blend_gate, pad_idx=-1):
        super().__init__()
        self.latent_dim = latent_dim
        self.pad_idx = pad_idx
        self.pool = MultiQueryPooling(latent_dim, int(num_queries))
        self.register_buffer("gate_task", torch.tensor(blend_gate, dtype=torch.bool), persistent=False)

    def forward(self, comp_idx, comp_frac, Z_species, task_id):
        mask = comp_idx == self.pad_idx
        Z = Z_species[comp_idx.clamp_min(0)]  # (B, L, latent_dim)
        frac = comp_frac.masked_fill(mask, 0.0)
        frac = frac / (frac.sum(dim=1, keepdim=True) + 1e-8)

        mean_pool = (Z * frac.unsqueeze(-1)).sum(dim=1)
        attn_pool = self.pool(Z, mask, frac)

        is_blend = ((~mask).sum(dim=1) > 1).float()
        scale = torch.where(self.gate_task[task_id], is_blend, torch.ones_like(is_blend))
        return mean_pool, attn_pool * scale.unsqueeze(1)
