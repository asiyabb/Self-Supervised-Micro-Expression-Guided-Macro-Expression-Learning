import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalViTEncoder(nn.Module):
    """
    Spatial Feature Extractor + Temporal Transformer.
    Used for Baseline and Ablation models.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.spatial_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embed_dim)
        )
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, batch_first=True),
            num_layers=2
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.spatial_backbone(x)
        feats = feats.view(B, T, self.embed_dim)
        encoded = self.temporal_transformer(feats)
        global_repr = encoded.mean(dim=1)
        return global_repr, encoded


class MicroGuidedMacroEncoder(nn.Module):
    """
    Proposed Method: Uses micro-clue temporal representations to guide
    macro-expression learning via Cross-Attention with robust temporal alignment.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.micro_encoder = TemporalViTEncoder(embed_dim)
        self.macro_encoder = TemporalViTEncoder(embed_dim)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, micro_x, macro_x):
        # 1. Extract micro-clue temporal sequence: shape (B, T_micro, D)
        with torch.no_grad():
            _, micro_seq = self.micro_encoder(micro_x)

        # 2. Extract macro temporal sequence: shape (B, T_macro, D)
        macro_global, macro_seq = self.macro_encoder(macro_x)

        # 3. Temporal Alignment: Interpolate micro_seq to match macro_seq temporal length (T_macro)
        # micro_seq shape: (B, T_micro, D). Permute to (B, D, T_micro) for 1D interpolation -> (B, D, T_macro) -> (B, T_macro, D)
        B, T_macro, D = macro_seq.shape
        micro_seq_aligned = F.interpolate(
            micro_seq.permute(0, 2, 1), 
            size=T_macro, 
            mode='linear', 
            align_corners=False
        ).permute(0, 2, 1)

        # 4. Cross-attention: Aligned micro clues guide Macro features
        guided_feats, _ = self.cross_attention(query=macro_seq, key=micro_seq_aligned, value=micro_seq_aligned)
        
        enhanced_macro_seq = self.projection(guided_feats + macro_seq)
        enhanced_macro_global = enhanced_macro_seq.mean(dim=1)

        return enhanced_macro_global, macro_global