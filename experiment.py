import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import LayerScale, Attention,VisionTransformer, init_weights_vit_timm
from timm.layers import DropPath, Mlp,PatchEmbed,LayerType, get_norm_layer, get_act_layer
from typing import Callable, Optional, Sequence, Tuple, Type, Union, List, Literal
from functools import partial
from timm.models._manipulate import checkpoint_seq 
from timm.models.vision_transformer import Block
import timm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)   # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)    # even index
        pe[0, :, 1::2] = torch.cos(position * div_term)    # odd index
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]     # add positional encoding to input tensor
        return x


class PatchFFT(nn.Module):
    """
    输入: 224×224 图像 → 分成 16×16 个 14×14 patch
    输出: [B, L=256, 336]  其中 L 可自适应池化到任意固定长度
    """
    def __init__(self, patch_size=14, L=256, channel_num = 3):
        super().__init__()
        self.patch_size = patch_size
        self.L = L
        self.channel_num = channel_num
        self.d_model = self.patch_size*(self.patch_size//2+1) #14*8=112
        # self.positional_embedding = PositionalEncoding(self.d_model, max_len=L)        # 单份



    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        return: [B*pn, C*d_model]
        """
        B, C, H, W = x.shape
        ps = self.patch_size
        pn = (H // ps) * (W // ps)
        # 1. RGB转灰度（保持维度）
        gray = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
        # gray shape: [B, 1, H, W]
        
        # 2. unfold
        x = gray.unfold(2, ps, ps).unfold(3, ps, ps)  # [B, 1, H/ps, W/ps, ps, ps]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, H/ps, W/ps, 1, ps, ps]
        x = x.view(-1, ps, ps)                    # [B*pn, ps,ps]

        # 2. FFT
        freq = torch.fft.rfft2(x, norm='ortho')
        freq = torch.fft.fftshift(freq)
        '''
        这里可以尝试用diagonal scan
        '''
        amp = torch.abs(freq)
        ang = torch.angle(freq)

        # 3. log
        amp = torch.log1p(1 + amp)              # [B*pn, ps, ps//2+1]
        ang = ang                               # angle 不加 log

        # 4. 先 reshape 成 [B*pn, d_model]
        amp = amp.view(B*pn, 1, -1)
        ang = ang.view(B*pn, 1, -1)
        '''
        先加position还是先norm?vit是先加可学习position再norm,这里和vit一样
        '''
        # 5. 先加位置编码
        # amp = self.positional_embedding(amp)    # [B*pn, 1, d_model]
        # ang = self.positional_embedding(ang)

        # 6. 再 norm（沿最后一维 d_model 计算均值方差）
        amp = (amp - amp.mean(dim=-1, keepdim=True)) / (amp.std(dim=-1, keepdim=True) + 1e-6)
        ang = (ang - ang.mean(dim=-1, keepdim=True)) / (ang.std(dim=-1, keepdim=True) + 1e-6)

        # 7. 三通道 cat
        amp = amp.flatten(1, 2).view(B,pn,self.d_model)                 # [B*pn, C*d_model]
        ang = ang.flatten(1, 2).view(B,pn,self.d_model) 
        '''
        尝试加入更多包含某一个patch的更大patch的fft特征
        '''
        return amp, ang

class CrossAttention(Attention):

    def __init__(
            self,
            dim: int,
            cond_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_norm,
            attn_drop,
            proj_drop,
            norm_layer,
        )
        del self.qkv
        self.q_proj = nn.Linear(dim, dim , bias=qkv_bias)
        self.kv_proj = nn.Linear(cond_dim,dim * 2,bias = qkv_bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        N_cond = cond.shape[1]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        kv = self.kv_proj(cond).reshape(B,N_cond,2,self.num_heads,self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FreqBlock(Block):
    def __init__(
            self,
            dim: int,
            cond_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        
        self.norm_amp = norm_layer(cond_dim)
        self.cross_attn_amp = CrossAttention(
            dim,
            cond_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls_amp = LayerScale(dim,init_values=init_values) if init_values else nn.Identity()
        self.drop_path_amp = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_ang = norm_layer(cond_dim)
        self.cross_attn_ang = CrossAttention(
            dim,
            cond_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.ls_ang = LayerScale(dim,init_values=init_values) if init_values else nn.Identity()
        self.drop_path_ang = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, cond_amp: torch.Tensor, cond_ang: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path_amp(self.ls_amp(self.cross_attn_amp(self.norm3(x),self.norm_amp(cond_amp))))
        x = x + self.drop_path_ang(self.ls_ang(self.cross_attn_ang(self.norm4(x),self.norm_ang(cond_ang))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionFrequencyTransformer(VisionTransformer):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU
        freq_dim = patch_size*(patch_size//2+1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            FreqBlock(
                dim=embed_dim,
                cond_dim=freq_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                init_values=init_values,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_layer=mlp_layer,
            ) for i in range(depth)
        ])
        self.patch_num = (img_size//patch_size)**2 if type(img_size) is int else (img_size[0]//patch_size)*(img_size[1]//patch_size)
        self.patchfft = PatchFFT(patch_size=patch_size,L = self.patch_num,channel_num = in_chans)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)
        amp, ang = self.patchfft(x) 
        # forward pass
        x = self.patch_embed(x)
        
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x,amp,ang)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x,n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        amp, ang = self.patchfft(x) 
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, amp, ang)
        else:
            for blk in self.blocks:
                x = blk(x, amp, ang)
        x = self.norm(x)
        return x
class FreqSemanticModel(nn.Module):
    def __init__(self):
        super().__init__()
        arch='vit_base_patch16_224'
        self.patch_feature_extractor = VisionFrequencyTransformer(global_pool='avg',)
        ret = self.patch_feature_extractor.load_state_dict(timm.create_model(arch, pretrained=True).state_dict(), strict=False)
        pretrained_set = set(self.patch_feature_extractor.state_dict().keys()) - set(ret.missing_keys)
        for n, p in self.patch_feature_extractor.named_parameters():
            p.requires_grad_(n not in pretrained_set)
        self.semantic_extractor = timm.create_model(arch, pretrained=True)
        for _,p in self.semantic_extractor.named_parameters():
            p.requires_grad_(False)
        self.classifier = Mlp(in_features=768*2,hidden_features=768,out_features=1)
        init_weights_vit_timm(self.classifier)
    def trainable_patch_keys(self):
        """真正需要被保存/加载的 patch 键名"""
        return {n for n, p in self.patch_feature_extractor.named_parameters()
                if p.requires_grad}

    def forward(self,x):
        patch_feature = self.patch_feature_extractor.forward_features(x)
        patch_feature = self.patch_feature_extractor.forward_head(patch_feature,pre_logits=True)
        semantic = self.semantic_extractor.forward_features(x)
        semantic = self.semantic_extractor.forward_head(semantic,pre_logits = True)
        logit = self.classifier(torch.cat([patch_feature,semantic],dim = -1))
        return logit
        
if __name__ == "__main__":
    
    model = FreqSemanticModel()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(out.shape)