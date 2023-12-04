import math
import time

import torch
import torch.nn as nn
from quantize.quantizer import UniformAffineQuantizer

# from .quant import *


DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# def quantize(x, scale, zero, maxq):
#     q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
#     return scale * (q - zero)


class GPTQ:
    def __init__(self, layer, quantizer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = quantizer

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01):
        W = self.layer.weight.data.clone()
        W = W.float()

        #print(f"W.shape={W.shape}")

        tick = time.time()


        self.quantizer.debug=100
        self.quantizer.calibration(W)
        self.quantizer.debug=0
        
        #print(f"During fasterquant, self.quantizer.scale: {self.quantizer.scale.shape}")
        #print(self.quantizer.scale)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H


        #print(f"self.quantizer.scale.size(1)={self.quantizer.scale.size(1)}")
        #print(f"self.columns={self.columns},blocksize={blocksize}")
        for i1 in range(0, self.columns, blocksize):
            
        
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]


            #print(f"i1={i1},i2={i2},count={count}")
            for i in range(count):
                #print(f"i={i}", end='')
                
                w = W1[:, i]
                d = Hinv1[i, i]

                # index = [i1 + i]
                dim = 1
                if self.quantizer.scale.size(dim) != 1:
                    # scale = torch.index_select(self.quantizer.scale, dim, index)
                    # round_zero_point = torch.index_select(
                    #     self.quantizer.round_zero_point, dim, index
                    # )
                    scale = self.quantizer.scale[:, i1 + i].unsqueeze(1)
                    
                    #print(f"scale.shape={scale.shape}")
                    
                    if self.quantizer.round_zero_point is not None:
                        round_zero_point = self.quantizer.round_zero_point[
                            :, i1 + i
                        ].unsqueeze(1)
                    else:
                        round_zero_point = None
                else:
                    scale = self.quantizer.scale
                    round_zero_point = self.quantizer.round_zero_point
                
                if(i1==0 and i==0):
                    self.quantizer.debug=2
                
                #print(f"i={i},w.shape={w.shape},w.unsqueeze(1).shape={w.unsqueeze(1).shape},scale.shape={scale.shape},round_zero_point.shape={round_zero_point.shape}")
                q = self.quantizer.fake_quant(
                    w.unsqueeze(1), scale, round_zero_point
                ).flatten()
                
                
                self.quantizer.debug=0
                
                # q = quantize(
                #     w.unsqueeze(1),
                #     self.quantizer.scale,
                #     self.quantizer.zero,
                #     self.quantizer.maxq,
                # ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        # print("time %.2f" % (time.time() - tick))
        # print("error", torch.sum(Losses).item())

        return (
            Q.reshape(self.layer.weight.shape),
            torch.sum(Losses).item(),
        )
        # if DEBUG:
        # print("quant error l2", torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        self.quantizer = None
        torch.cuda.empty_cache()
