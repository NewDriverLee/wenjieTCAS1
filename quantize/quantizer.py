import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np

CLIPMIN = 1e-4


class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1
        
        if(n_bits<=4):
            self.qmin_double = -(2 ** (n_bits*2 - 1))
            self.qmax_double = 2 ** (n_bits*2 - 1) - 1
        else:
            self.qmin_double = self.qmin
            self.qmax_double = self.qmax
        
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method

        self.mode = "calibration"
        self.enable = True
        self.recorded_quant_input=None
        
        self.debug=0
        self.topk_en=0
        self.dynamic_quant=0
        self.first_several_group_en = 0

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1

    def quant(self, x, scale, round_zero_point, uint=False):
        x_int = (x / scale).round_()
        if uint:
            x_int = x_int.add_(round_zero_point - self.qmin)
            x_int = x_int.clamp_(0, self.qmax - self.qmin).to(torch.uint8)
            return x_int
        else:
            if not self.symmetric:
                x_int = x_int.add_(round_zero_point)
            x_int = x_int.clamp_(self.qmin, self.qmax).to(torch.int8)
            return x_int

    def dequant(self, x_int, scale, round_zero_point, uint=False):
        if not x_int.dtype == scale.dtype:
            x_int = x_int.to(scale.dtype)
        if uint:
            if round_zero_point is not None:
                x_int = x_int.sub_(round_zero_point - self.qmin)
            x_dequant = x_int.mul_(scale)
            return x_dequant
        else:
            if round_zero_point is not None:
                x_int = x_int.sub_(round_zero_point)
            x_dequant = x_int.mul_(scale)
            return x_dequant

    def fake_quant(self, x, scale, round_zero_point):
        # start quantization
        
        #if(self.debug==1):
        #    print(f"Before quantization: self.n_bits={self.n_bits}")
        #    print(f"x: {x.shape}")
        #    print(x)
        
        
        if(self.dynamic_quant==1):
            #x/scale + round_zero_point > self.qmax or x/scale + round_zero_point < self.qmin
            #x > scale*(self.qmax-round_zero_point) or x < scale*(self.qmin-round_zero_point)
            
            #mask_max = (x > scale*(self.qmax-round_zero_point)).long()
            #mask_min = (x < scale*(self.qmin-round_zero_point)).long()
            #mask_max_min = mask_max + mask_min
            #mask_low_bit = torch.ones_like(mask_max_min)
            #mask_low_bit = mask_low_bit - mask_max_min
            
            
            x_int = (x/ scale).round()
            if round_zero_point is not None:
                x_int = x_int.add(round_zero_point)
            
            #x_int_max_min = torch.clamp(x_int,self.topk_qmin, self.topk_qmax)
            #x_int_max_min = x_int_max_min*mask_max_min
            #
            #x_int_low_bit = torch.clamp(x_int,self.qmin, self.qmax)
            #x_int_low_bit = x_int_low_bit*mask_low_bit
            #
            #x_dequant = x_int_max_min + x_int_low_bit
            
            x_int_clamp = torch.clamp(x_int,self.topk_qmin, self.topk_qmax)
            x_dequant = x_int_clamp
            
            ######### Count for sparsity ##########
            cnt_tensor = (x_dequant > self.qmax).long() + (x_dequant < self.qmin).long()
            self.extra_element = self.extra_element + torch.count_nonzero(cnt_tensor)
            self.num_sample = self.num_sample + 1
            ######### Count for sparsity ##########
            
            if round_zero_point is not None:
                x_dequant = x_dequant.sub_(round_zero_point)
            x_dequant = x_dequant.mul_(scale)
            
            return x_dequant
        
        
        x_int = (x / scale).round_()
        if round_zero_point is not None:
            x_int = x_int.add_(round_zero_point)
        #x_int_clamp = x_int.clamp_(self.qmin, self.qmax) # original version
        x_int_clamp = torch.clamp(x_int,self.qmin, self.qmax)
        
        
            
        
        #if(self.debug==1):
        #    print(f"After quantization: self.n_bits={self.n_bits}")
        #    print(f"x_int: {x_int.shape}")
        #    print(x_int)
        
        #if(self.debug==2):
        #    print(f"During GPTQ: self.n_bits={self.n_bits}")
        #    print(f"x_int: {x_int.shape}")
        #    print(x_int)
        #    print("\n")
        
        x_dequant = x_int_clamp
        if(self.topk_en==1  and self.first_several_group_en==0):  # if self.first_several_group_en==1, topk will not be invoked
            #x_dequant[:,:,self.topk_idx]=x_int[:,:,self.topk_idx]
            #print("!!!!")
            if(x_dequant.dim()==3):
                x_dequant[:,:,self.topk_idx]=torch.clamp(x_int[:,:,self.topk_idx],self.topk_qmin, self.topk_qmax)
            else:
                x_dequant[:,self.topk_idx]=torch.clamp(x_int[:,self.topk_idx],self.topk_qmin, self.topk_qmax)
        
        if(self.first_several_group_en==1):
            #print(f"self.idx_first_several_group={self.idx_first_several_group}")
            #print(f"self.n_bits={self.n_bits}, self.qmin_double={self.qmin_double},self.qmax_double={self.qmax_double}")
            if(x_dequant.dim()==3):
                x_dequant[:,:,self.idx_first_several_group]=torch.clamp(x_int[:,:,self.idx_first_several_group],self.qmin_double, self.qmax_double)
            else:
                x_dequant[:,self.idx_first_several_group]=torch.clamp(x_int[:,self.idx_first_several_group],self.qmin_double, self.qmax_double)
        
        
        if round_zero_point is not None:
            x_dequant = x_dequant.sub_(round_zero_point)
        x_dequant = x_dequant.mul_(scale)
        
        # float point
        #if(self.first_several_group_en==1):
        #    if(x_dequant.dim()==3):
        #        x_dequant[:,:,self.idx_first_several_group]=x[:,:,self.idx_first_several_group]
        #    else:
        #        x_dequant[:,self.idx_first_several_group]=x[:,self.idx_first_several_group]
        
        
        # Use float-point for outlier
        #if(self.topk_en==1):
        #    #print(f"x_dequant.shape={x_dequant.shape}")
        #    x_dequant[:,:,self.topk_idx]=x[:,:,self.topk_idx]
        
        return x_dequant

    def forward(self, x: torch.Tensor):

        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(255).round_().div_(255) #original
            #return x.mul_(256).round_().div_(256) #8b
            #return x.mul_(4096).round_().div_(4096) #12b
            
        if self.dynamic:
            if self.dynamic_method == "per_token":
                self.per_token_dynamic_calibration(x)
            elif self.dynamic_method == "per_cluster":
                self.calibration(x)
        elif self.mode == "calibration":
            self.calibration(x)
            return x

        # start quantization
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        scale = (xmax - xmin) / (2**self.n_bits)
        # scale = (xmax - xmin) * (2**-self.n_bits)
        self.scale = scale.clamp_(min=CLIPMIN, max=1e4)
        # zero_point = (xmax + xmin) * (-0.5 / scale)
        zero_point = -(xmax + xmin) / (2 * scale)
        self.round_zero_point = zero_point.clamp_(min=-1e4, max=1e4).round_()

    def calibration(self, x: torch.Tensor):
        reduce_axes = [
            _
            for _ in range(x.ndim)
            if _ not in self.per_channel_axes and _ != self.cluster_dim
        ]
        
        #if(self.debug==100):
        #    print(f"reduce_axes={reduce_axes}, self.metric={self.metric}")
        
        if self.metric in ["layer_mse", "minmax", "ema_minmax"]:
            scale, zero_point = self.minmax_calibration(x, reduce_axes)
            
            #print("In calibration:")
            #print(f"self.metric: {self.metric}")
            #print(f"scale: {scale.shape}")
            #print(f"zero_point: {zero_point.shape}")
            
        elif self.metric == "mse":
            scale, zero_point = self.mse_calibration(x, reduce_axes)

        del self.scale
        self.register_buffer("scale", scale)
        # self.scale = scale
        if zero_point is not None:
            zero_point.clamp_(min=-1e4, max=1e4)
            del self.zero_point, self.round_zero_point
            self.register_buffer("zero_point", zero_point)
            self.register_buffer("round_zero_point", zero_point.round())
            # self.zero_point = zero_point
            # self.round_zero_point = self.zero_point.round()

        # debug
        if not self.dynamic:
            if torch.isinf(self.scale).any() or torch.isnan(self.scale).any():
                breakpoint()
            if self.zero_point is not None:
                if (
                    torch.isinf(self.round_zero_point).any()
                    or torch.isnan(self.round_zero_point).any()
                ):
                    breakpoint()
        if x.size(0) == 1:
            self.pack_dim = 1

    def minmax_calibration(self, x, reduce_axes):
        # minmax
        #print(f"self.topk_en={self.topk_en}")
        if(self.topk_en==1 and self.first_several_group_en==0): # if self.first_several_group_en==1, topk will not be invoked
            x_tmp = torch.empty_like(x)
            x_tmp.copy_(x).to(x.device)
            #print("!!!!")
            #print(f"x_tmp.shape={x_tmp.shape}")
            #print(f"self.topk_idx={self.topk_idx}")
            if(x_tmp.dim()==3):
                x_tmp[:,:,self.topk_idx_calibrate]=0
            else:
                x_tmp[:,self.topk_idx_calibrate]=0
            
            
                        
            #x_tmp=x
        else:
            x_tmp=x
        
        #if(self.debug==100):
        #    print(f"self.symmetric={self.symmetric}, len(reduce_axes)={len(reduce_axes)}, self.dynamic={self.dynamic}, self.cluster_dim={self.cluster_dim}, self.cached_xmax={self.cached_xmax}")
        
        if self.symmetric:
            #print("symmetric!!!!!!!!!")
            if len(reduce_axes):
                abs_max = x_tmp.abs().amax(reduce_axes, keepdim=True)
            else:
                abs_max = x_tmp.abs()
            scale = abs_max / (2 ** (self.n_bits - 1))
            if self.cluster_dim is not None:
                # for cluster quantization
                st = 0
                for count in self.cluster_counts:
                    part_scale = torch.narrow(scale, self.cluster_dim, st, count)
                    cluster_max = part_scale.amax(
                        self.cluster_dim, keepdim=True
                    )  # 不该保持维度
                    scale.narrow(self.cluster_dim, st, count).copy_(cluster_max)
                    st += count
            scale.clamp_(min=CLIPMIN, max=1e4)
            zero_point = None
        else:
            if len(reduce_axes):
                xmin = x_tmp.amin(reduce_axes, keepdim=True)
                xmax = x_tmp.amax(reduce_axes, keepdim=True)
            else:
                xmin = x_tmp.clone()
                xmax = x_tmp.clone()
                
            #if(self.topk_en==1):
            #    print(f"xmax: {xmax.shape}")
            #    print(f"xmax[0][112]={xmax[0][112]}, xmin[0][112]={xmin[0][112]}")
            #    print(xmax)
            #    print(f"torch.argmax(xmax,dim=2)={torch.argmax(xmax,dim=2)}, torch.argmax(xmax,dim=2)={torch.argmin(xmin,dim=2)}")
            
            
            if not self.dynamic:
                if self.cached_xmax is not None:
                    if self.metric == "minmax":
                        xmax = torch.max(self.cached_xmax, xmax)
                        xmin = torch.min(self.cached_xmin, xmin)
                    if self.metric == "ema_minmax":
                        xmax = self.cached_xmax * 0.99 + xmax * 0.01
                        xmin = self.cached_xmin * 0.99 + xmin * 0.01
                self.cached_xmax = xmax
                self.cached_xmin = xmin
                
                #if(self.debug==100):
                #    print(f"self.cached_xmax.shape={self.cached_xmax.shape}, xmax={xmax.shape}")
                
            if self.cluster_dim is not None:
                # for cluster quantization
                
                #print(f"for cluster quantization: self.cluster_dim={self.cluster_dim}")
                #print(f"self.cluster_counts={self.cluster_counts}")
                #print(f"xmax.shape: {xmax.shape}")
                #print(xmax)
                
                st = 0
                #if(self.debug==100):
                #    print(f"self.cluster_counts={self.cluster_counts}")
                for count in self.cluster_counts:
                    part_xmin = torch.narrow(xmin, self.cluster_dim, st, count)
                    part_xmax = torch.narrow(xmax, self.cluster_dim, st, count)
                    
                    #if(count==337):
                    #    print(f"part_xmax: {part_xmax.shape}")
                    #    print(part_xmax)
                    
                    cluster_xmin = part_xmin.amin(self.cluster_dim, keepdim=True)
                    cluster_xmax = part_xmax.amax(self.cluster_dim, keepdim=True)
                    
                    #if(self.debug==100):
                    #    print(f"part_xmin.shape={part_xmin.shape}, cluster_xmin.shape={cluster_xmin.shape}")
                    
                    xmin.narrow(self.cluster_dim, st, count).copy_(cluster_xmin)
                    xmax.narrow(self.cluster_dim, st, count).copy_(cluster_xmax)
                    
                    #if(self.debug==100):
                    #    print("cluster_xmin:")
                    #    print(cluster_xmin)
                    #    print("xmin:")
                    #    print(xmin)
                    
                    st += count
            scale = (xmax - xmin) * (2**-self.n_bits)
            
            
            
            if(self.first_several_group_en==1):
                #print(f"Before, scale={scale}")
                if(scale.dim()==3):
                    if(self.n_bits<=4):
                        scale[:,:,self.idx_first_several_group] = scale[:,:,self.idx_first_several_group] / (2**-self.n_bits) * (2**(-self.n_bits*2))
                else:
                    if(self.n_bits<=4):
                        scale[:,self.idx_first_several_group] = scale[:,self.idx_first_several_group] / (2**-self.n_bits) * (2**(-self.n_bits*2))
                #print(f"After, scale={scale}")    
            
            #if(self.topk_en==1):
            #    print(f"scale[0][112]={scale[0][112]}")
            #if(self.debug==100):
            #    print(f"scale.shape={scale.shape}")
            #if(scale.ndim==3):
            #    for ii in range(xmax.shape[-1]):
            #        print(f"{ii}:{scale[0][0][ii]} ",end='')
            #    print('\n\n')
            
            
            #if(torch.count_nonzero(scale<CLIPMIN)!=0):
            #    print(f"self.n_bits={self.n_bits}")
            #    print(f"scale={scale}")
            #    if(scale.dim()==3):
            #        indices = torch.nonzero(scale[0][0]<CLIPMIN)
            #        print(f"indices[0]={indices[0]}, scale[0][0][indices[0]]={scale[0][0][indices[0]]}")
            #    else:
            #        indices = torch.nonzero(scale[0]<CLIPMIN)
            #        print(f"indices[0]={indices[0]}, scale[0][indices[0]]={scale[0][indices[0]]}")
            #    #print(f"indices={indices}")
            #    #print(f"scale[indices[0]]={scale[indices[0]]}")
                
            scale.clamp_(min=CLIPMIN, max=1e4)
            #scale.clamp_(min=1e-10, max=1e4) #lwj
            
            
            #if(self.topk_en==1):
            #    print(f"after clamp, scale[0][112]={scale[0][112]}")
            zero_point = (xmax + xmin) * (-0.5 / scale)
            
                
            
            
            
            
            if(self.topk_en==1  and self.first_several_group_en==0):
                del x_tmp
            
            return scale, zero_point

    def mse_calibration(self, x, reduce_axes):
        # mse
        if self.symmetric:
            abs_max = x.abs().amax(reduce_axes, keepdim=True)
            min_loss = None
            best_scale = None
            for t in torch.linspace(0.1, 1, steps=50):
                t = abs_max * t
                scale = t / (2 ** (self.n_bits - 1))
                x_sim = self.fake_quant(x, scale, None)
                loss = F.mse_loss(x, x_sim)
                if min_loss is None or min_loss > loss:
                    min_loss = loss
                    best_scale = scale
            scale = best_scale
            zero_point = None
        else:
            xmin = x.amin(reduce_axes, keepdim=True)
            xmax = x.amax(reduce_axes, keepdim=True)
            x_center = (xmin + xmax) / 2
            min_loss = None
            best_scale = None
            best_zero_point = None
            for i in torch.linspace(0, 1, steps=30):
                t_left = xmin * i + x_center * (1 - i)
                for j in torch.linspace(0.1, 1, steps=30):
                    t_right = t_left * j + xmax * (1 - j)
                    if (t_left == t_right).any():
                        continue
                    scale = (t_right - t_left) / (2**self.n_bits)
                    zero_point = -((t_right + t_left) / 2 / scale)
                    x_sim = self.fake_quant(x, scale, zero_point)
                    loss = F.mse_loss(x, x_sim, reduce=False).mean(
                        reduce_axes, keepdim=True
                    )
                    if min_loss is None:
                        min_loss = loss
                        best_scale = scale
                        best_zero_point = zero_point
                    else:
                        mask = min_loss > loss
                        best_scale[mask] = scale
                        best_zero_point[mask] = zero_point
            scale = best_scale
            zero_point = best_zero_point
        scale.clamp_(min=1e-6, max=1e4)
        return scale, zero_point

    def layer_mse_param_search_update(
        self, qlayer, a_quantizers, inps, outs, attention_mask, steps=10
    ):
        assert not self.dynamic and (
            not self.symmetric
        ), "only support static asymmetric in layer_mse_calibration now"

        # prepare_layer_mse_calibration
        self.qlayer = qlayer
        self.attention_mask = attention_mask
        self.quantizers_nbit_dict = {}
        if self.n_bits == 16:
            return
        for name, quantizer in a_quantizers.items():
            self.quantizers_nbit_dict[quantizer] = quantizer.n_bits
            quantizer.n_bits = 16

        self.n_bits = self.quantizers_nbit_dict[self]
        xmax = self.cached_xmax
        xmin = self.cached_xmin
        xrange = xmax - xmin
        step_interval = xrange / steps

        print("=== start layer_mse_param_search_update ===")
        dev = self.scale.device
        attention_mask = self.attention_mask.to(dev)

        nsamples = inps.size(0)
        best_mse = 0
        for j in range(nsamples):
            sim_outs = qlayer(
                inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask
            )[0]
            best_mse += F.mse_loss(sim_outs[0], outs[j].to(dev))
        print(f"best_mse {best_mse}")

        if self.cluster_dim is not None:
            # for cluster quantization
            st = 0
            for count in self.cluster_counts:
                part_xmin = torch.narrow(xmin, self.cluster_dim, st, count)
                part_xmax = torch.narrow(xmax, self.cluster_dim, st, count)
                part_step_interval = torch.narrow(
                    step_interval, self.cluster_dim, st, count
                )
                for i in range(steps):
                    # left
                    new_part_xmin = part_xmin + part_step_interval
                    new_part_scale = (part_xmax - new_part_xmin) * (2**-self.n_bits)
                    new_part_scale.clamp_(min=CLIPMIN, max=1e4)
                    self.scale.narrow(self.cluster_dim, st, count).copy_(new_part_scale)
                    new_part_zero_point = (part_xmax + new_part_xmin) * (
                        -0.5 / new_part_scale
                    )
                    new_part_zero_point.clamp_(min=-1e4, max=1e4).round_()
                    self.round_zero_point.narrow(self.cluster_dim, st, count).copy_(
                        new_part_zero_point
                    )
                    mse = 0
                    for j in range(nsamples):
                        sim_outs = qlayer(
                            inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask
                        )[0]
                        mse += F.mse_loss(sim_outs[0], outs[j].to(dev))
                    print(f"mse {mse}")
                    if mse < best_mse:
                        # accept
                        best_mse = mse
                        part_xmin = new_part_xmin
                        print(f"update min, {best_mse}")

                    # right
                    new_part_xmax = part_xmax - part_step_interval
                    new_part_scale = (new_part_xmax - part_xmin) * (2**-self.n_bits)
                    new_part_scale.clamp_(min=CLIPMIN, max=1e4)
                    self.scale.narrow(self.cluster_dim, st, count).copy_(new_part_scale)
                    new_part_zero_point = (new_part_xmax + part_xmin) * (
                        -0.5 / new_part_scale
                    )
                    new_part_zero_point.clamp_(min=-1e4, max=1e4).round_()
                    self.round_zero_point.narrow(self.cluster_dim, st, count).copy_(
                        new_part_zero_point
                    )
                    mse = 0
                    for j in range(nsamples):
                        sim_outs = qlayer(
                            inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask
                        )[0]
                        mse += F.mse_loss(sim_outs[0], outs[j].to(dev))
                    if mse < best_mse:
                        # accept
                        best_mse = mse
                        part_xmax = new_part_xmax
                        print(f"update max, {best_mse}")
                st += count

        # free_layer_mse_calibration
        self.qlayer = None
        self.attention_mask = None
        for quantizer, n_bits in self.quantizers_nbit_dict.items():
            quantizer.n_bits = n_bits
        self.quantizers_nbit_dict = None

    def set_calibration_mode(self):
        self.mode = "calibration"

    def set_eval_mode(self):
        self.mode = "eval"

    def free(self):
        del self.cached_xmin
        del self.cached_xmax
        del self.recorded_quant_input


if __name__ == "__main__":
    # test cluster quant
    q = UniformAffineQuantizer(8, False, [0])
    q.cluster_dim = 1
    q.cluster_counts = [3, 7]
    x = torch.rand([4, 10])
    q.calibration(x)
    q.forward(x)
    print(q.scale)
