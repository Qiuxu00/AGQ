"""Fused unpack + per-output-channel scale, preserving EXP009 BF16 values."""
import torch,triton
import triton.language as tl

@triton.jit
def decode_kernel(P,S,Z,O, N:tl.constexpr,K:tl.constexpr,BITS:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=idx//K;col=idx%K;valid=idx<N*K
    byte=col*BITS//8;shift=col*BITS%8
    stride=(K*BITS+7)//8
    lo=tl.load(P+row*stride+byte,mask=valid,other=0).to(tl.int32)
    if BITS==3:
        hi=tl.load(P+row*stride+byte+1,mask=valid & (byte+1<stride),other=0).to(tl.int32)
        code=((lo | (hi<<8))>>shift)&((1<<BITS)-1)
    else:
        code=(lo>>shift)&((1<<BITS)-1)
    scale=tl.load(S+row,mask=valid,other=0).to(tl.float32)
    if BITS==1:
        value=(code.to(tl.float32)*2.-1.)*scale
    else:
        zero=tl.load(Z+row,mask=valid,other=0).to(tl.float32)
        value=(code.to(tl.float32)-zero)*scale
    tl.store(O+idx,value.to(tl.bfloat16),mask=valid)

def decode(module,block=1024,warps=4,out=None):
    n,k=module.out_features,module.in_features
    if out is None:out=torch.empty((n,k),dtype=torch.bfloat16,device=module.packed.device)
    decode_kernel[(triton.cdiv(n*k,block),)](module.packed,module.scale,module.zero_or_reserved,out,n,k,module.bits,block,num_warps=warps,enable_fp_fusion=False)
    return out
