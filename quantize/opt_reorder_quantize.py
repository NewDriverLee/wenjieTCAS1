import torch
import torch.nn as nn
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.reorder_layer_norm import ReorderLayerNorm
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.quant_transformer_layer import quant_layer
from quantize.reorder_utils import (
    tensor_calc_reorder_index,
    ic_maxmin_dict,
    oc_maxmin_dict,
    oc_maxmin_dict_debug,
    layer_i0max_hook,
    layer_omax_hook,
)
import numpy as np

R_DEBUG_BIT = 0
DEBUG_BREAK_LAYER = -1


def R1_reorder(layer_norm, qproj, kproj, vproj, index, counts):
    layer_norm.register_buffer("reorder_index", index)
    layer_norm.out_quantizer.cluster_dim = 2
    layer_norm.out_quantizer.cluster_counts = counts
    if R_DEBUG_BIT:
        layer_norm.out_quantizer.change_n_bits(R_DEBUG_BIT)

    #print("kproj.weight.data:")
    #print(kproj.weight.data)

    qproj.weight.data = torch.index_select(qproj.weight.data, 1, index)
    qproj.set_ic_cluster_counts(counts, a_dim=None)


    #print(f"index={index}")
    #print(f"kproj.weight.data[0][2020]: {kproj.weight.data[0][2020]}")
    #print(f"kproj.weight.data[0][1990]: {kproj.weight.data[0][1990]}")
    #print(f"kproj.weight.data[0][1999]: {kproj.weight.data[0][1999]}")
    kproj.weight.data = torch.index_select(kproj.weight.data, 1, index)
    
    #print("kproj.weight.data:")
    #print(kproj.weight.data)
    
    kproj.set_ic_cluster_counts(counts, a_dim=None)
    vproj.weight.data = torch.index_select(vproj.weight.data, 1, index)
    vproj.set_ic_cluster_counts(counts, a_dim=None)


def R2_reorder(qproj, kproj, qkt_matmul, index, counts):
    qproj.weight.data = torch.index_select(qproj.weight.data, 0, index)
    qproj.bias.data = torch.index_select(qproj.bias.data, 0, index)
    kproj.weight.data = torch.index_select(kproj.weight.data, 0, index)
    kproj.bias.data = torch.index_select(kproj.bias.data, 0, index)

    qkt_matmul.set_ic_cluster_counts(counts, x1_dim=2, x2_dim=2)
    if R_DEBUG_BIT:
        qkt_matmul.x1_quantizer.change_n_bits(R_DEBUG_BIT)
        qkt_matmul.x2_quantizer.change_n_bits(R_DEBUG_BIT)


def R3_reorder(vproj, pv_matmul, out_proj, index, counts):
    vproj.weight.data = torch.index_select(vproj.weight.data, 0, index)
    vproj.bias.data = torch.index_select(vproj.bias.data, 0, index)
    pv_matmul.set_ic_cluster_counts(counts, cluster_x1=False)
    out_proj.weight.data = torch.index_select(out_proj.weight.data, 1, index)
    out_proj.set_ic_cluster_counts(counts)
    if R_DEBUG_BIT:
        pv_matmul.x2_quantizer.change_n_bits(R_DEBUG_BIT)
        out_proj.act_quantizer.change_n_bits(R_DEBUG_BIT)


def R4_reorder(layer_norm, fc1, index, counts):
    layer_norm.register_buffer("reorder_index", index)

    layer_norm.out_quantizer.cluster_dim = 1
    layer_norm.out_quantizer.cluster_counts = counts

    fc1.weight.data = torch.index_select(fc1.weight.data, 1, index)
    fc1.set_ic_cluster_counts(counts, a_dim=None)
    if R_DEBUG_BIT:
        layer_norm.out_quantizer.change_n_bits(R_DEBUG_BIT)


def R5_reorder(fc1, fc2, index, counts):
    fc1.weight.data = torch.index_select(fc1.weight.data, 0, index)
    fc1.bias.data = torch.index_select(fc1.bias.data, 0, index)

    fc2.weight.data = torch.index_select(fc2.weight.data, 1, index)
    fc2.set_ic_cluster_counts(counts, a_dim=1)
    if R_DEBUG_BIT:
        fc2.act_quantizer.change_n_bits(R_DEBUG_BIT)


@torch.no_grad()
def opt_reorder_quantize(
    lm,
    args,
    dataloader,
    n_clusters={"R1": 4, "R2": 4, "R3": 4, "R4": 32, "R5": 4},
    reorder="12345",
):
    print("Starting ...")

    model = lm.model
    dev = lm.device

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    print(f"dev={dev}")

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    # only catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            #print("cache[i]:")
            #print(cache["i"])
            #print(f"inp={inp}")
            
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError
    
    
    #print("layers[0]:")
    #print(layers[0])
    #
    #print("model:")
    #print(model)
    
    layers[0] = Catcher(layers[0])

    #print("layers[0]:")
    #print(layers[0])
    #print("layers[1]:")
    #print(layers[1])
    #
    #print("model:")
    #print(model)

    #print(f"inps({inps.shape})={inps}")

    for batch in dataloader:
        #print("cache[i]:")
        #print(cache["i"])
        if cache["i"] >= args.nsamples:
            break
        try:
            model(batch[0].to(dev))

        except ValueError:
            pass

    #print("model:")
    #print(model)

    #print(f"inps({inps.shape})={inps}")

    

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    enable_R1 = True if "1" in reorder else False
    enable_R2 = True if "2" in reorder else False
    enable_R3 = True if "3" in reorder else False
    enable_R4 = True if "4" in reorder else False
    enable_R5 = True if "5" in reorder else False
    print(f"Ready for reorder {reorder}.")

    for i in range(len(layers)):
        #if(i==2):
        #    break
        
        if i == DEBUG_BREAK_LAYER:
            break
        print(f"=== Start quantize layer {i} ===")
        
        #print(f"inps({inps.shape})={inps}")
        
        layer = layers[i].to(dev)
        
        #print("Original layer:")
        #print(layer)
        
        #print("\n\n")
        #print(f"layer.self_attn.k_proj.weight: {layer.self_attn.k_proj.weight.shape}")
        #print(layer.self_attn.k_proj.weight)
        #print("\n\n")
        
        #print("oc_maxmin_dict 1:")
        #print(oc_maxmin_dict)
        
        qlayer = QuantOPTDecoderLayer(lm.model.config, layer, args)

        #print("qlayer before reodering 1:")
        #print(qlayer)
        
        #print("\n\n")
        #print(f"qlayer.self_attn.k_proj.weight: {qlayer.self_attn.k_proj.weight.shape}")
        #print(qlayer.self_attn.k_proj.weight)
        #print("\n\n")
        
        
        #print("\n\n")
        #print(f"qlayer.self_attn.k_proj.weight[2020]: {qlayer.self_attn.k_proj.weight[2020]}")
        #print(f"qlayer.self_attn.k_proj.weight[1990]: {qlayer.self_attn.k_proj.weight[1990]}")
        #print(f"qlayer.self_attn.k_proj.weight[1999]: {qlayer.self_attn.k_proj.weight[1999]}")
        #print("\n\n")
        #
        #print("oc_maxmin_dict 2:")
        #print(oc_maxmin_dict)

        # register hook for data
        handlers = []
        for name, module in layer.named_modules():
            if (
                enable_R1
                and isinstance(module, nn.LayerNorm)
                and "attn_layer_norm" in name
            ):
                # print(f"register R1 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if (
                enable_R2
                and isinstance(module, nn.Linear)
                #and ("q_proj" in name or "k_proj" in name)
                and ("q_proj" in name or "k_proj" in name or "v_proj" in name)
            ):
                # print(f"register R2 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R3 and isinstance(module, nn.Linear) and "out_proj" in name:
                # print(f"register R3 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if (
                enable_R4
                and isinstance(module, nn.LayerNorm)
                and "final_layer_norm" in name
            ):
                # print(f"register R4 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R5 and isinstance(module, nn.Linear) and "fc2" in name:
                # print(f"register R5 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)


        #print("qlayer before reodering 2:")
        #print(qlayer)
        
        #print("\n\n")
        #print(f"qlayer.self_attn.k_proj.weight: {qlayer.self_attn.k_proj.weight.shape}")
        #print(qlayer.self_attn.k_proj.weight)
        #print("\n\n")
        #
        #print("oc_maxmin_dict 3:")
        #print(oc_maxmin_dict)

        # inference to collect data for reordering
        #print(f"args.nsamples={args.nsamples}")
        #print(f"inps({inps.shape})={inps}")
        
        for j in range(args.nsamples):
        #for j in range(1):
            #print(f"j={j}")
            outs[j] = layer(
                inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev)
            )[0]
            
        #print("oc_maxmin_dict 4:")
        #print(oc_maxmin_dict)
        #
        #print("ic_maxmin_dict:")
        #print(ic_maxmin_dict)
        #
        #print(f"oc_maxmin_dict[self_attn_layer_norm]: {oc_maxmin_dict['self_attn_layer_norm'][0].shape}, {oc_maxmin_dict['self_attn_layer_norm'][1].shape}")
        #print(f"oc_maxmin_dict[self_attn.q_proj]: {oc_maxmin_dict['self_attn.q_proj'][0].shape}, {oc_maxmin_dict['self_attn.q_proj'][1].shape}")
        #print(f"oc_maxmin_dict[self_attn.k_proj]: {oc_maxmin_dict['self_attn.k_proj'][0].shape}, {oc_maxmin_dict['self_attn.k_proj'][1].shape}")
        #print(f"oc_maxmin_dict[final_layer_norm]: {oc_maxmin_dict['final_layer_norm'][0].shape}, {oc_maxmin_dict['final_layer_norm'][1].shape}")
        #
        #print(f"ic_maxmin_dict[self_attn.out_proj]: {ic_maxmin_dict['self_attn.out_proj'][0].shape}, {ic_maxmin_dict['self_attn.out_proj'][1].shape}")
        #print(f"ic_maxmin_dict[fc2]: {ic_maxmin_dict['fc2'][0].shape}, {ic_maxmin_dict['fc2'][1].shape}")
        #
        
        '''
        ################# self_attn_layer_norm #########################
        #fp=open("LN_abs_max_min_div2_sorted.txt",'a')
        #tmp_activation=(torch.abs(oc_maxmin_dict['self_attn_layer_norm'][0])+torch.abs(oc_maxmin_dict['self_attn_layer_norm'][1]))/2
        #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        #fp=open("LN_mixed_calibrate.txt",'a')
        #sorted_tmp_activation = (torch.abs(oc_maxmin_dict['self_attn_layer_norm'][0])+torch.abs(oc_maxmin_dict['self_attn_layer_norm'][1]))/2
        ##print(f"sorted_idx={sorted_idx}")
        #for ii in range(sorted_tmp_activation.shape[-1]):
        #    fp.write(str(sorted_tmp_activation[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        #fp=open("LN_abs_max_min_div2_idx.txt",'a')
        #for ii in range(sorted_idx.shape[-1]):
        #    fp.write(str(sorted_idx[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        
        ################# q_proj k_proj #########################
        #fp=open("q_proj_abs_max_min_div2_sorted.txt",'a')
        #tmp_activation=(torch.abs(oc_maxmin_dict['self_attn.q_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.q_proj'][1]))/2
        #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        #fp=open("q_proj_mixed_calibrate.txt",'a')
        #sorted_tmp_activation = (torch.abs(oc_maxmin_dict['self_attn.q_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.q_proj'][1]))/2
        #for ii in range(sorted_tmp_activation.shape[-1]):
        #    fp.write(str(sorted_tmp_activation[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        #fp=open("q_proj_abs_max_min_div2_idx.txt",'a')
        #for ii in range(sorted_idx.shape[-1]):
        #    fp.write(str(sorted_idx[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        #fp=open("k_proj_abs_max_min_div2_sorted.txt",'a')
        #tmp_activation=(torch.abs(oc_maxmin_dict['self_attn.k_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.k_proj'][1]))/2
        #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        fp=open("k_proj_mixed_calibrate.txt",'a')
        sorted_tmp_activation = (torch.abs(oc_maxmin_dict['self_attn.k_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.k_proj'][1]))/2
        for ii in range(sorted_tmp_activation.shape[-1]):
            fp.write(str(sorted_tmp_activation[ii].item())+' ')
        fp.write('\n')
        fp.close()
        #
        #fp=open("k_proj_abs_max_min_div2_idx.txt",'a')
        #for ii in range(sorted_idx.shape[-1]):
        #    fp.write(str(sorted_idx[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        #
        #fp=open("v_proj_abs_max_min_div2_sorted.txt",'a')
        #tmp_activation=(torch.abs(oc_maxmin_dict['self_attn.v_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.v_proj'][1]))/2
        #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        fp=open("v_proj_mixed_calibrate.txt",'a')
        sorted_tmp_activation = (torch.abs(oc_maxmin_dict['self_attn.v_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.v_proj'][1]))/2
        for ii in range(sorted_tmp_activation.shape[-1]):
            fp.write(str(sorted_tmp_activation[ii].item())+' ')
        fp.write('\n')
        fp.close()
        #
        #fp=open("v_proj_abs_max_min_div2_idx.txt",'a')
        #for ii in range(sorted_idx.shape[-1]):
        #    fp.write(str(sorted_idx[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        ################# final_layer_norm #########################
        #fp=open("final_layer_norm_abs_max_min_div2_sorted.txt",'a')
        #tmp_activation=(torch.abs(oc_maxmin_dict['final_layer_norm'][0])+torch.abs(oc_maxmin_dict['final_layer_norm'][1]))/2
        #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        #fp = open("final_layer_norm_mixed_calibrate.txt",'a')
        #sorted_tmp_activation = (torch.abs(oc_maxmin_dict['final_layer_norm'][0])+torch.abs(oc_maxmin_dict['final_layer_norm'][1]))/2
        #for ii in range(sorted_tmp_activation.shape[-1]):
        #    fp.write(str(sorted_tmp_activation[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        #fp=open("final_layer_norm_abs_max_min_div2_idx.txt",'a')
        #for ii in range(sorted_idx.shape[-1]):
        #    fp.write(str(sorted_idx[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        ################# self_attn.out_proj #########################
        #fp=open("out_proj_abs_max_min_div2_sorted.txt",'a')
        #tmp_activation=(torch.abs(ic_maxmin_dict['self_attn.out_proj'][0])+torch.abs(ic_maxmin_dict['self_attn.out_proj'][1]))/2
        #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        #fp = open("out_proj_mixed_calibrate.txt",'a')
        #sorted_tmp_activation = (torch.abs(ic_maxmin_dict['self_attn.out_proj'][0])+torch.abs(ic_maxmin_dict['self_attn.out_proj'][1]))/2
        #for ii in range(sorted_tmp_activation.shape[-1]):
        #    fp.write(str(sorted_tmp_activation[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        #fp=open("out_proj_abs_max_min_div2_idx.txt",'a')
        #for ii in range(sorted_idx.shape[-1]):
        #    fp.write(str(sorted_idx[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        ################# fc2 #########################
        #fp=open("fc2_abs_max_min_div2_sorted.txt",'a')
        #tmp_activation=(torch.abs(ic_maxmin_dict['fc2'][0])+torch.abs(ic_maxmin_dict['fc2'][1]))/2
        #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        #fp = open("fc2_mixed_calibrate.txt",'a')
        #sorted_tmp_activation = (torch.abs(ic_maxmin_dict['fc2'][0])+torch.abs(ic_maxmin_dict['fc2'][1]))/2
        #for ii in range(sorted_tmp_activation.shape[-1]):
        #    fp.write(str(sorted_tmp_activation[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        
        #fp=open("fc2_abs_max_min_div2_idx.txt",'a')
        #for ii in range(sorted_idx.shape[-1]):
        #    fp.write(str(sorted_idx[ii].item())+' ')
        #fp.write('\n')
        #fp.close()
        '''
        
        for handler in handlers:
            handler.remove()
    
    
        #print("qlayer before reodering 3:")
        ##print(qlayer)
        #
        #print("\n\n")
        #print(f"qlayer.self_attn.k_proj.weight: {qlayer.self_attn.k_proj.weight.shape}")
        #print(qlayer.self_attn.k_proj.weight)
        #print("\n\n")
    
        
        ############################################### TopK for ours ###################################################
        ###################  TopK for self_attn_layer_norm ###########################
        tmp_activation=(torch.abs(oc_maxmin_dict['self_attn_layer_norm'][0])+torch.abs(oc_maxmin_dict['self_attn_layer_norm'][1]))/2
        #tmp_activation=torch.max(torch.abs(oc_maxmin_dict['self_attn_layer_norm'][0]),torch.abs(oc_maxmin_dict['self_attn_layer_norm'][1]))
        sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        R1_index = sorted_idx
        
        topk_idx=sorted_idx[0:args.topk_num]
        #topk_idx=sorted_idx[0:128]
        #topk_idx=torch.arange(args.topk_num) # Since the activations will be sorted later
        
        if(args.group128 == 1):
            tmp_activation=(torch.abs(oc_maxmin_dict['self_attn_layer_norm'][0])+torch.abs(oc_maxmin_dict['self_attn_layer_norm'][1]))/2
            topk_idx=torch.empty(0, dtype=topk_idx.dtype).to(topk_idx.device)
            for iii in range(int(layer.embed_dim / qlayer.self_attn.head_dim)):
            #for iii in range(4):
                #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*qlayer.self_attn.head_dim:(iii+1)*qlayer.self_attn.head_dim], descending=True)
                #tmp_tensor = sorted_idx[0:args.topk_num]+(iii*qlayer.self_attn.head_dim)
                
                tmp_tensor=torch.arange(args.topk_num).to(topk_idx.device) +(iii*qlayer.self_attn.head_dim) # Since the activations will be sorted later
                topk_idx=torch.cat((topk_idx,tmp_tensor))
        
        #print(f"topk_idx.shape: {topk_idx.shape}")
        #print(topk_idx)
        qlayer.self_attn_layer_norm.out_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.self_attn_layer_norm.out_quantizer.register_buffer('topk_idx_calibrate', topk_idx)
        qlayer.self_attn_layer_norm.out_quantizer.topk_en=1
        print(f"qlayer.self_attn_layer_norm.out_quantizer.n_bits={qlayer.self_attn_layer_norm.out_quantizer.n_bits}")
        qlayer.self_attn_layer_norm.out_quantizer.topk_qmax=2 ** (2*qlayer.self_attn_layer_norm.out_quantizer.n_bits - 1) - 1
        qlayer.self_attn_layer_norm.out_quantizer.topk_qmin=-(2 ** (2*qlayer.self_attn_layer_norm.out_quantizer.n_bits - 1))
        #qlayer.self_attn_layer_norm.out_quantizer.dynamic_quant=1
        
        
        #if(args.group128 == 1):
        #    topk_idx = torch.arange(qlayer.self_attn.head_dim * 2).to(topk_idx.device)
        #    qlayer.self_attn_layer_norm.out_quantizer.first_several_group_en=1
        #    qlayer.self_attn_layer_norm.out_quantizer.register_buffer('idx_first_several_group', topk_idx)
        #    #print(f"topk_idx={topk_idx}")
        
        ###################  TopK for final_layer_norm ###########################
        tmp_activation=(torch.abs(oc_maxmin_dict['final_layer_norm'][0])+torch.abs(oc_maxmin_dict['final_layer_norm'][1]))/2
        #tmp_activation=torch.max(torch.abs(oc_maxmin_dict['final_layer_norm'][0]),torch.abs(oc_maxmin_dict['final_layer_norm'][1]))
        sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        R4_index = sorted_idx
        
        #topk_idx=sorted_idx[0:args.topk_num_final_layer_norm]
        #topk_idx=sorted_idx[0:128]
        #topk_idx=torch.arange(args.topk_num_final_layer_norm) # Since the activations will be sorted later
        
        if(args.group128 == 1):
            tmp_activation=(torch.abs(oc_maxmin_dict['final_layer_norm'][0])+torch.abs(oc_maxmin_dict['final_layer_norm'][1]))/2
            topk_idx=torch.empty(0, dtype=topk_idx.dtype).to(topk_idx.device)
            for iii in range(int(layer.embed_dim / qlayer.self_attn.head_dim)):
            #for iii in range(4):
                #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*qlayer.self_attn.head_dim:(iii+1)*qlayer.self_attn.head_dim], descending=True)
                #tmp_tensor = sorted_idx[0:args.topk_num_final_layer_norm]+(iii*qlayer.self_attn.head_dim)
                
                tmp_tensor=torch.arange(args.topk_num_final_layer_norm).to(topk_idx.device) +(iii*qlayer.self_attn.head_dim) # Since the activations will be sorted later
                topk_idx=torch.cat((topk_idx,tmp_tensor))
        
        qlayer.final_layer_norm.out_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.final_layer_norm.out_quantizer.register_buffer('topk_idx_calibrate', topk_idx)
        qlayer.final_layer_norm.out_quantizer.topk_en=1
        print(f"qlayer.final_layer_norm.out_quantizer.n_bits={qlayer.final_layer_norm.out_quantizer.n_bits}")
        qlayer.final_layer_norm.out_quantizer.topk_qmax=2 ** (2*qlayer.final_layer_norm.out_quantizer.n_bits - 1) - 1
        qlayer.final_layer_norm.out_quantizer.topk_qmin=-(2 ** (2*qlayer.final_layer_norm.out_quantizer.n_bits - 1))
        #qlayer.final_layer_norm.out_quantizer.dynamic_quant=1
        
        #if(args.group128 == 1):
        #    topk_idx = torch.arange(qlayer.self_attn.head_dim * 2).to(topk_idx.device)
        #    qlayer.final_layer_norm.out_quantizer.first_several_group_en=1
        #    qlayer.final_layer_norm.out_quantizer.register_buffer('idx_first_several_group', topk_idx)
        #    #print(f"topk_idx={topk_idx}")
            
            
        
        ###################  TopK for fc2 ###########################
        tmp_activation=(torch.abs(ic_maxmin_dict['fc2'][0])+torch.abs(ic_maxmin_dict['fc2'][1]))/2
        #tmp_activation=torch.max(torch.abs(ic_maxmin_dict['fc2'][0]),torch.abs(ic_maxmin_dict['fc2'][1]))
        sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        R5_index = sorted_idx
        
        #topk_idx=sorted_idx[0:args.topk_num_fc2]
        #topk_idx=sorted_idx[0:128]
        #topk_idx=torch.arange(args.topk_num_fc2) # Since the activations will be sorted later
        
        if(args.group128 == 1):
            tmp_activation=(torch.abs(ic_maxmin_dict['fc2'][0])+torch.abs(ic_maxmin_dict['fc2'][1]))/2
            topk_idx=torch.empty(0, dtype=topk_idx.dtype).to(topk_idx.device)
            for iii in range(int(4*layer.embed_dim / qlayer.self_attn.head_dim)):
            #for iii in range(4):
                #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*qlayer.self_attn.head_dim:(iii+1)*qlayer.self_attn.head_dim], descending=True)
                #tmp_tensor = sorted_idx[0:args.topk_num_fc2]+(iii*qlayer.self_attn.head_dim)
                
                tmp_tensor=torch.arange(args.topk_num_fc2).to(topk_idx.device) +(iii*qlayer.self_attn.head_dim) # Since the activations will be sorted later
                topk_idx=torch.cat((topk_idx,tmp_tensor))
                
            ##for iii in range(int(layer.embed_dim / qlayer.self_attn.head_dim)):
            #    #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*qlayer.self_attn.head_dim*4:(iii+1)*qlayer.self_attn.head_dim*4], descending=True)
            #    #tmp_tensor = sorted_idx[0:args.topk_num_fc2]+(iii*qlayer.self_attn.head_dim*4)
            #    
            #    tmp_tensor=torch.arange(args.topk_num_fc2).to(topk_idx.device) +(iii*qlayer.self_attn.head_dim) # Since the activations will be sorted later
            #    topk_idx=torch.cat((topk_idx,tmp_tensor))
                
        
        qlayer.fc2.act_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.fc2.act_quantizer.register_buffer('topk_idx_calibrate', topk_idx)
        qlayer.fc2.act_quantizer.topk_en=1
        print(f"qlayer.fc2.act_quantizer.n_bits={qlayer.fc2.act_quantizer.n_bits}")
        qlayer.fc2.act_quantizer.topk_qmax=2 ** (2*qlayer.fc2.act_quantizer.n_bits - 1) - 1
        qlayer.fc2.act_quantizer.topk_qmin=-(2 ** (2*qlayer.fc2.act_quantizer.n_bits - 1))
        #qlayer.fc2.act_quantizer.dynamic_quant=1
        
        #if(args.group128 == 1):
        #    topk_idx = torch.arange(qlayer.self_attn.head_dim * 2).to(topk_idx.device)
        #    qlayer.fc2.act_quantizer.first_several_group_en=1
        #    qlayer.fc2.act_quantizer.register_buffer('idx_first_several_group', topk_idx)
        #    #print(f"topk_idx={topk_idx}")
        
        
        ###################  TopK for self_attn.out_proj ###########################
        tmp_activation=(torch.abs(ic_maxmin_dict['self_attn.out_proj'][0])+torch.abs(ic_maxmin_dict['self_attn.out_proj'][1]))/2
        #tmp_activation=torch.max(torch.abs(ic_maxmin_dict['self_attn.out_proj'][0]),torch.abs(ic_maxmin_dict['self_attn.out_proj'][1]))
        
        topk_idx=torch.empty(0, dtype=topk_idx.dtype).to(topk_idx.device)
        
        
        for iii in range(qlayer.self_attn.num_heads):
            sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*qlayer.self_attn.head_dim:(iii+1)*qlayer.self_attn.head_dim], descending=True)
            tmp_tensor = sorted_idx[0:args.topk_num_out_proj_head]+(iii*qlayer.self_attn.head_dim)
            
            topk_idx=torch.cat((topk_idx,tmp_tensor))
            
        #for iii in range(int(layer.embed_dim / 16)):
        #    sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*16:(iii+1)*16], descending=True)
        #    tmp_tensor = sorted_idx[0:args.topk_num_out_proj_head]+(iii*16)
        #    topk_idx=torch.cat((topk_idx,tmp_tensor))
        
        #topk_idx=torch.empty(0, dtype=topk_idx.dtype).to(topk_idx.device)
        #for iii in range(qlayer.self_attn.num_heads):
        #    tmp_tensor=torch.arange(args.topk_num_out_proj_head)+(iii*128)
        #    topk_idx=torch.cat((topk_idx,tmp_tensor))
            
        qlayer.self_attn.out_proj.act_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.self_attn.out_proj.act_quantizer.register_buffer('topk_idx_calibrate', topk_idx)
        qlayer.self_attn.out_proj.act_quantizer.topk_en=1
        print(f"qlayer.self_attn.out_proj.act_quantizer.n_bits={qlayer.self_attn.out_proj.act_quantizer.n_bits}")
        qlayer.self_attn.out_proj.act_quantizer.topk_qmax=2 ** (2*qlayer.self_attn.out_proj.act_quantizer.n_bits - 1) - 1
        qlayer.self_attn.out_proj.act_quantizer.topk_qmin=-(2 ** (2*qlayer.self_attn.out_proj.act_quantizer.n_bits - 1))
        #qlayer.self_attn.out_proj.act_quantizer.dynamic_quant=1
        
        ###################  TopK for self_attn.q_proj ###########################
        tmp_activation=(torch.abs(oc_maxmin_dict['self_attn.q_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.q_proj'][1]))/2
        #tmp_activation=torch.max(torch.abs(oc_maxmin_dict['self_attn.q_proj'][0]),torch.abs(oc_maxmin_dict['self_attn.q_proj'][1]))
        
        topk_idx=torch.empty(0, dtype=topk_idx.dtype).to(topk_idx.device)
        for iii in range(qlayer.self_attn.num_heads):
            sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*qlayer.self_attn.head_dim:(iii+1)*qlayer.self_attn.head_dim], descending=True)
            tmp_tensor = sorted_idx[0:args.topk_num_q_proj_head]+(iii*qlayer.self_attn.head_dim)
            
            topk_idx=torch.cat((topk_idx,tmp_tensor))
            
        #for iii in range(int(layer.embed_dim / 16)):
        #    sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*16:(iii+1)*16], descending=True)
        #    tmp_tensor = sorted_idx[0:args.topk_num_q_proj_head]+(iii*16)
        #    topk_idx=torch.cat((topk_idx,tmp_tensor))
            
        
        #topk_idx=torch.empty(0, dtype=topk_idx.dtype).to(topk_idx.device)
        #for iii in range(qlayer.self_attn.num_heads):
        #    tmp_tensor=torch.arange(args.topk_num_q_proj_head)+(iii*128)
        #    topk_idx=torch.cat((topk_idx,tmp_tensor))
            
        qlayer.self_attn.qkt_matmul.x1_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.self_attn.qkt_matmul.x1_quantizer.register_buffer('topk_idx_calibrate', topk_idx)
        qlayer.self_attn.qkt_matmul.x1_quantizer.topk_en=1
        print(f"qlayer.self_attn.qkt_matmul.x1_quantizer.n_bits={qlayer.self_attn.qkt_matmul.x1_quantizer.n_bits}")
        qlayer.self_attn.qkt_matmul.x1_quantizer.topk_qmax=2 ** (2*qlayer.self_attn.qkt_matmul.x1_quantizer.n_bits - 1) - 1
        qlayer.self_attn.qkt_matmul.x1_quantizer.topk_qmin=-(2 ** (2*qlayer.self_attn.qkt_matmul.x1_quantizer.n_bits - 1))
        #qlayer.self_attn.qkt_matmul.x1_quantizer.dynamic_quant=1
        

        if enable_R1:
            #print("Before kmeans:")
            #print(oc_maxmin_dict[f"self_attn_layer_norm"])
        
            #feature_max, feature_min = oc_maxmin_dict[f"self_attn_layer_norm"]
            #R1_index, counts = tensor_calc_reorder_index(
            #    feature_max, feature_min, n_clusters["R1"]
            #)
            #R1_counts = counts
            
            #print("After kmeans:")
            #print(oc_maxmin_dict[f"self_attn_layer_norm"])
            
            #tmp_activation=(torch.abs(oc_maxmin_dict['self_attn_layer_norm'][0])+torch.abs(oc_maxmin_dict['self_attn_layer_norm'][1]))/2
            #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
            #R1_index = sorted_idx

            if(args.group128 == 1):
                #counts = np.full(int(layer.embed_dim / qlayer.self_attn.head_dim), qlayer.self_attn.head_dim)
                
                #counts = np.full(int(layer.embed_dim / 256), 256)
                counts = np.full(int(layer.embed_dim / args.group_size), args.group_size)
            
            print(f"R1_index {R1_index.shape}: {R1_index}")
            
            #for ii in range(len(R1_index)):
            #    print(f"{ii}:{R1_index[ii]} ",end='')
            #print('\n')
            
            print(f"R1 counts {counts.shape}: {counts}")
            R1_reorder(
                qlayer.self_attn_layer_norm,
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.v_proj,
                R1_index,
                counts,
            )
            
        #print("qlayer before reodering 4:")
        #print(qlayer)
        
        #print("\n\n")
        #print(f"qlayer.self_attn.k_proj.weight: {qlayer.self_attn.k_proj.weight.shape}")
        #print(qlayer.self_attn.k_proj.weight)
        #print("\n\n")

        if enable_R2:
            qmax, qmin = oc_maxmin_dict[f"self_attn.q_proj"]
            kmax, kmin = oc_maxmin_dict[f"self_attn.k_proj"]
            R2_index, counts = tensor_calc_reorder_index(
                [qmax, kmax], [qmin, kmin], n_clusters["R2"], qlayer.self_attn.num_heads
            )
            
            R2_counts = counts
            
            # Determine R2 using only Q's calibration results
            #tmp_activation=(torch.abs(oc_maxmin_dict['self_attn.q_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.q_proj'][1]))/2
            #R2_index=torch.empty(0, dtype=R2_index.dtype).to(R2_index.device)
            #for iii in range(qlayer.self_attn.num_heads):
            #    sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*128:(iii+1)*128], descending=True)
            #    #print(f"iii={iii}, sorted_idx={sorted_idx}")
            #    sorted_idx = sorted_idx+(iii*128)
            #    #print(f"sorted_idx={sorted_idx}")
            #    R2_index=torch.cat((R2_index,sorted_idx))
            
            #if(args.group128 == 1):
            #    counts = np.full(int(layer.embed_dim / 16), 16)
            
            if((args.group128 == 1) and (qlayer.self_attn.head_dim > args.group_size)):
                counts = np.full(int(layer.embed_dim / args.group_size), args.group_size)
            
            print(f"R2_index {R2_index.shape}: {R2_index}")
            print(f"R2 counts {counts.shape}: {counts}")
            R2_reorder(
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.qkt_matmul,
                R2_index,
                counts,
            )

        #print("qlayer before reodering 5:")
        ##print(qlayer)
        #
        #print("\n\n")
        #print(f"qlayer.self_attn.k_proj.weight: {qlayer.self_attn.k_proj.weight.shape}")
        #print(qlayer.self_attn.k_proj.weight)
        #print("\n\n")

        if enable_R3:
            feature_max, feature_min = ic_maxmin_dict[f"self_attn.out_proj"]
            R3_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R3"], qlayer.self_attn.num_heads
            )
            R3_counts = counts
            
            #tmp_activation=(torch.abs(ic_maxmin_dict['self_attn.out_proj'][0])+torch.abs(ic_maxmin_dict['self_attn.out_proj'][1]))/2
            #R3_index=torch.empty(0, dtype=R3_index.dtype).to(R3_index.device)
            #for iii in range(qlayer.self_attn.num_heads):
            #    sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation[iii*128:(iii+1)*128], descending=True)
            #    #print(f"iii={iii}, sorted_idx={sorted_idx}")
            #    sorted_idx = sorted_idx+(iii*128)
            #    #print(f"sorted_idx={sorted_idx}")
            #    R3_index=torch.cat((R3_index,sorted_idx))
            
            #if(args.group128 == 1):
            #    counts = np.full(int(layer.embed_dim / 16), 16)
            
            if((args.group128 == 1) and (qlayer.self_attn.head_dim > args.group_size)):
                counts = np.full(int(layer.embed_dim / args.group_size), args.group_size)
            
            print(f"R3_index {R3_index.shape}: {R3_index}")
            print(f"R3 counts {counts.shape}: {counts}")
            R3_reorder(
                qlayer.self_attn.v_proj,
                qlayer.self_attn.pv_matmul,
                qlayer.self_attn.out_proj,
                R3_index,
                counts,
            )

        if enable_R4:
            #feature_max, feature_min = oc_maxmin_dict[f"final_layer_norm"]
            #R4_index, counts = tensor_calc_reorder_index(
            #    feature_max, feature_min, n_clusters["R4"]
            #)
            #R4_counts = counts
            
            
            #fp=open('R4_counts_opt13b.txt','a')
            #for ii in range(len(R4_counts)):
            #    fp.write(str(R4_counts[ii])+' ')
            #fp.write('\n')
            #fp.close()
            
            #tmp_activation=(torch.abs(oc_maxmin_dict['final_layer_norm'][0])+torch.abs(oc_maxmin_dict['final_layer_norm'][1]))/2
            #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
            #R4_index = sorted_idx

            if(args.group128 == 1):
                #counts = np.full(int(layer.embed_dim / qlayer.self_attn.head_dim), qlayer.self_attn.head_dim)
                
                #counts = np.full(int(layer.embed_dim / 256), 256)
                counts = np.full(int(layer.embed_dim / args.group_size), args.group_size)
            
            print(f"R4_index {R4_index.shape}: {R4_index}")
            print(f"R4 counts {counts.shape}: {counts}")
            R4_reorder(
                qlayer.final_layer_norm,
                qlayer.fc1,
                R4_index,
                counts,
            )

        if enable_R5:
            #feature_max, feature_min = ic_maxmin_dict[f"fc2"]
            #R5_index, counts = tensor_calc_reorder_index(
            #    feature_max, feature_min, n_clusters["R5"]
            #)
            #R5_counts = counts
            
            #tmp_activation=(torch.abs(ic_maxmin_dict['fc2'][0])+torch.abs(ic_maxmin_dict['fc2'][1]))/2
            #sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
            #R5_index = sorted_idx

            if(args.group128 == 1):
                #counts = np.full(int(4*layer.embed_dim / qlayer.self_attn.head_dim), qlayer.self_attn.head_dim)
                #counts = np.full(int(layer.embed_dim / qlayer.self_attn.head_dim), qlayer.self_attn.head_dim*4)
                
                #counts = np.full(int(4*layer.embed_dim / 256), 256)
                counts = np.full(int(4*layer.embed_dim / args.group_size), args.group_size)
            
            print(f"R5_index {R5_index.shape}: {R5_index}")
            print(f"R5 counts {counts.shape}: {counts}")
            R5_reorder(
                qlayer.fc1,
                qlayer.fc2,
                R5_index,
                counts,
            )



        '''
        ############################################### TopK for RPTQ ###################################################
        ####################  TopK for self_attn_layer_norm ###########################
        tmp_activation=(torch.abs(oc_maxmin_dict['self_attn_layer_norm'][0])+torch.abs(oc_maxmin_dict['self_attn_layer_norm'][1]))/2
        #print(f"before selection: {tmp_activation}")
        tmp_activation = torch.index_select(tmp_activation, 0, R1_index)
        #print(f"after selection: {tmp_activation}")
        
        sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        #print(f"sorted_tmp_activation={sorted_tmp_activation}")
        #print(f"sorted_idx={sorted_idx}")
        
        topk_idx=sorted_idx[0:args.topk_num]
        #print(f"topk_idx={topk_idx}")
        #topk_idx=torch.arange(args.topk_num) # Since the activations will be sorted later
        
        qlayer.self_attn_layer_norm.out_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.self_attn_layer_norm.out_quantizer.topk_en=1
        print(f"qlayer.self_attn_layer_norm.out_quantizer.n_bits={qlayer.self_attn_layer_norm.out_quantizer.n_bits}")
        qlayer.self_attn_layer_norm.out_quantizer.topk_qmax=2 ** (2*qlayer.self_attn_layer_norm.out_quantizer.n_bits - 1) - 1
        qlayer.self_attn_layer_norm.out_quantizer.topk_qmin=-(2 ** (2*qlayer.self_attn_layer_norm.out_quantizer.n_bits - 1))
        #qlayer.self_attn_layer_norm.out_quantizer.dynamic_quant=1
        
        #calibration
        # Check if the columns of a cluster are all selected by the topk method
        topk_idx_calibrate = torch.empty_like(topk_idx)
        topk_idx_calibrate.copy_(topk_idx).to(topk_idx.device)
        topk_idx_calibrate,_=torch.sort(topk_idx_calibrate, descending=False)
        #print(f"R1 topk_idx_calibrate={topk_idx_calibrate}")
        col_idx=0
        for ii in range(len(R1_counts)):
            #print(f"ii={ii}, R1_counts[ii]={R1_counts[ii]}")
            flag=0
            for jj in range(R1_counts[ii]):
                #if(ii==0):
                #    print(f"col_idx={col_idx}")
                if col_idx not in topk_idx_calibrate:
                    flag=1
                col_idx = col_idx + 1
                
            if(flag==0): # In this case, all the columns in the cluster should be recovered
                end_position = (torch.nonzero(topk_idx_calibrate == (col_idx-1)))[0].item()
                #print(f"col_idx-1={col_idx-1}, end_position={end_position}")
                topk_idx_calibrate = torch.cat([topk_idx_calibrate[:end_position-R1_counts[ii]+1], topk_idx_calibrate[end_position+1:]])
                #print(f"topk_idx_calibrate={topk_idx_calibrate}")
        qlayer.self_attn_layer_norm.out_quantizer.register_buffer('topk_idx_calibrate', topk_idx_calibrate)
        ##print(f"topk_idx_calibrate={topk_idx_calibrate}")
        
        
        ###################  TopK for final_layer_norm ###########################
        tmp_activation=(torch.abs(oc_maxmin_dict['final_layer_norm'][0])+torch.abs(oc_maxmin_dict['final_layer_norm'][1]))/2
        #print(f"before selection: {tmp_activation}")
        tmp_activation = torch.index_select(tmp_activation, 0, R4_index)
        #print(f"after selection: {tmp_activation}")
        sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        #print(f"sorted_tmp_activation={sorted_tmp_activation}")
        #print(f"sorted_idx={sorted_idx}")
        
        topk_idx=sorted_idx[0:args.topk_num_final_layer_norm]
        #print(f"topk_idx={topk_idx}")
        #topk_idx=torch.arange(args.topk_num_final_layer_norm) # Since the activations will be sorted later
 
        qlayer.final_layer_norm.out_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.final_layer_norm.out_quantizer.topk_en=1
        print(f"qlayer.final_layer_norm.out_quantizer.n_bits={qlayer.final_layer_norm.out_quantizer.n_bits}")
        qlayer.final_layer_norm.out_quantizer.topk_qmax=2 ** (2*qlayer.final_layer_norm.out_quantizer.n_bits - 1) - 1
        qlayer.final_layer_norm.out_quantizer.topk_qmin=-(2 ** (2*qlayer.final_layer_norm.out_quantizer.n_bits - 1))
        #qlayer.final_layer_norm.out_quantizer.dynamic_quant=1
        
        #calibration
        # Check if the columns of a cluster are all selected by the topk method
        topk_idx_calibrate = torch.empty_like(topk_idx)
        topk_idx_calibrate.copy_(topk_idx).to(topk_idx.device)
        topk_idx_calibrate,_=torch.sort(topk_idx_calibrate, descending=False)
        #print(f"R4 topk_idx_calibrate={topk_idx_calibrate}")
        col_idx=0
        for ii in range(len(R4_counts)):
            #print(f"ii={ii}, R4_counts[ii]={R1_counts[ii]}")
            flag=0
            for jj in range(R4_counts[ii]):
                if col_idx not in topk_idx_calibrate:
                    flag=1
                col_idx = col_idx + 1
                
            if(flag==0): # In this case, all the columns in the cluster should be recovered
                end_position = (torch.nonzero(topk_idx_calibrate == (col_idx-1)))[0].item()
                #print(f"col_idx-1={col_idx-1}, end_position={end_position}")
                topk_idx_calibrate = torch.cat([topk_idx_calibrate[:end_position-R4_counts[ii]+1], topk_idx_calibrate[end_position+1:]])
                #print(f"topk_idx_calibrate={topk_idx_calibrate}")
        qlayer.final_layer_norm.out_quantizer.register_buffer('topk_idx_calibrate', topk_idx_calibrate)
        ##print(f"topk_idx_calibrate={topk_idx_calibrate}")
        
        
        
        ###################  TopK for fc2 ###########################
        tmp_activation=(torch.abs(ic_maxmin_dict['fc2'][0])+torch.abs(ic_maxmin_dict['fc2'][1]))/2
        tmp_activation = torch.index_select(tmp_activation, 0, R5_index)
        sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        topk_idx=sorted_idx[0:args.topk_num_fc2]
        #print(f"topk_idx={topk_idx}")
        
        #topk_idx=torch.arange(args.topk_num_fc2) # Since the activations will be sorted later
        
        qlayer.fc2.act_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.fc2.act_quantizer.topk_en=1
        print(f"qlayer.fc2.act_quantizer.n_bits={qlayer.fc2.act_quantizer.n_bits}")
        qlayer.fc2.act_quantizer.topk_qmax=2 ** (2*qlayer.fc2.act_quantizer.n_bits - 1) - 1
        qlayer.fc2.act_quantizer.topk_qmin=-(2 ** (2*qlayer.fc2.act_quantizer.n_bits - 1))
        #qlayer.fc2.act_quantizer.dynamic_quant=1
        
        #calibration
        # Check if the columns of a cluster are all selected by the topk method
        topk_idx_calibrate = torch.empty_like(topk_idx)
        topk_idx_calibrate.copy_(topk_idx).to(topk_idx.device)
        topk_idx_calibrate,_=torch.sort(topk_idx_calibrate, descending=False)
        #print(f"R5 topk_idx_calibrate={topk_idx_calibrate}")
        col_idx=0
        for ii in range(len(R5_counts)):
            #print(f"ii={ii}")
            flag=0
            for jj in range(R5_counts[ii]):
                if col_idx not in topk_idx_calibrate:
                    flag=1
                col_idx = col_idx + 1
                
            if(flag==0): # In this case, all the columns in the cluster should be recovered
                end_position = (torch.nonzero(topk_idx_calibrate == (col_idx-1)))[0].item()
                topk_idx_calibrate = torch.cat([topk_idx_calibrate[:end_position-R5_counts[ii]+1], topk_idx_calibrate[end_position+1:]])
        qlayer.fc2.act_quantizer.register_buffer('topk_idx_calibrate', topk_idx_calibrate)
        ##print(f"topk_idx_calibrate={topk_idx_calibrate}")
        
        
        
        
        ###################  TopK for self_attn.out_proj ###########################
        tmp_activation=(torch.abs(ic_maxmin_dict['self_attn.out_proj'][0])+torch.abs(ic_maxmin_dict['self_attn.out_proj'][1]))/2
        tmp_activation = torch.index_select(tmp_activation, 0, R3_index)
        sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        topk_idx=sorted_idx[0:args.topk_num_out_proj_head*qlayer.self_attn.num_heads]
        #print(f"topk_idx={topk_idx}")
            
        qlayer.self_attn.out_proj.act_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.self_attn.out_proj.act_quantizer.topk_en=1
        print(f"qlayer.self_attn.out_proj.act_quantizer.n_bits={qlayer.self_attn.out_proj.act_quantizer.n_bits}")
        qlayer.self_attn.out_proj.act_quantizer.topk_qmax=2 ** (2*qlayer.self_attn.out_proj.act_quantizer.n_bits - 1) - 1
        qlayer.self_attn.out_proj.act_quantizer.topk_qmin=-(2 ** (2*qlayer.self_attn.out_proj.act_quantizer.n_bits - 1))
        #qlayer.self_attn.out_proj.act_quantizer.dynamic_quant=1
        
        #calibration
        # Check if the columns of a cluster are all selected by the topk method
        topk_idx_calibrate = torch.empty_like(topk_idx)
        topk_idx_calibrate.copy_(topk_idx).to(topk_idx.device)
        topk_idx_calibrate,_=torch.sort(topk_idx_calibrate, descending=False)
        #print(f"R3 topk_idx_calibrate={topk_idx_calibrate}")
        col_idx=0
        for ii in range(len(R3_counts)):
            #print(f"ii={ii}")
            flag=0
            for jj in range(R3_counts[ii]):
                if col_idx not in topk_idx_calibrate:
                    flag=1
                col_idx = col_idx + 1
                
            if(flag==0): # In this case, all the columns in the cluster should be recovered
                end_position = (torch.nonzero(topk_idx_calibrate == (col_idx-1)))[0].item()
                topk_idx_calibrate = torch.cat([topk_idx_calibrate[:end_position-R3_counts[ii]+1], topk_idx_calibrate[end_position+1:]])
        qlayer.self_attn.out_proj.act_quantizer.register_buffer('topk_idx_calibrate', topk_idx_calibrate)
        ##print(f"topk_idx_calibrate={topk_idx_calibrate}")
        
        
        
        
        ###################  TopK for self_attn.q_proj ###########################
        tmp_activation=(torch.abs(oc_maxmin_dict['self_attn.q_proj'][0])+torch.abs(oc_maxmin_dict['self_attn.q_proj'][1]))/2
        
        tmp_activation = torch.index_select(tmp_activation, 0, R2_index)
        sorted_tmp_activation, sorted_idx = torch.sort(tmp_activation, descending=True)
        
        topk_idx=sorted_idx[0:args.topk_num_q_proj_head*qlayer.self_attn.num_heads]
        #print(f"topk_idx={topk_idx}")
        
        qlayer.self_attn.qkt_matmul.x1_quantizer.register_buffer('topk_idx', topk_idx)
        qlayer.self_attn.qkt_matmul.x1_quantizer.topk_en=1
        print(f"qlayer.self_attn.qkt_matmul.x1_quantizer.n_bits={qlayer.self_attn.qkt_matmul.x1_quantizer.n_bits}")
        qlayer.self_attn.qkt_matmul.x1_quantizer.topk_qmax=2 ** (2*qlayer.self_attn.qkt_matmul.x1_quantizer.n_bits - 1) - 1
        qlayer.self_attn.qkt_matmul.x1_quantizer.topk_qmin=-(2 ** (2*qlayer.self_attn.qkt_matmul.x1_quantizer.n_bits - 1))
        #qlayer.self_attn.qkt_matmul.x1_quantizer.dynamic_quant=1
        
        #calibration
        # Check if the columns of a cluster are all selected by the topk method
        topk_idx_calibrate = torch.empty_like(topk_idx)
        topk_idx_calibrate.copy_(topk_idx).to(topk_idx.device)
        topk_idx_calibrate,_=torch.sort(topk_idx_calibrate, descending=False)
        #print(f"R2 topk_idx_calibrate={topk_idx_calibrate}")
        col_idx=0
        for ii in range(len(R2_counts)):
            #print(f"ii={ii}")
            flag=0
            for jj in range(R2_counts[ii]):
                if col_idx not in topk_idx_calibrate:
                    flag=1
                col_idx = col_idx + 1
                
            if(flag==0): # In this case, all the columns in the cluster should be recovered
                end_position = (torch.nonzero(topk_idx_calibrate == (col_idx-1)))[0].item()
                topk_idx_calibrate = torch.cat([topk_idx_calibrate[:end_position-R2_counts[ii]+1], topk_idx_calibrate[end_position+1:]])
        qlayer.self_attn.qkt_matmul.x1_quantizer.register_buffer('topk_idx_calibrate', topk_idx_calibrate)
        ##print(f"topk_idx_calibrate={topk_idx_calibrate}")
        '''
        





        #print("qlayer after reodering:")
        ##print(qlayer)
        #
        #print("\n\n")
        #print(f"qlayer.self_attn.k_proj.weight: {qlayer.self_attn.k_proj.weight.shape}")
        #print(qlayer.self_attn.k_proj.weight)
        #print("\n\n")

        

        outs = quant_layer(qlayer, args, outs, inps, attention_mask, dev)


        #print(f"qlayer.self_attn_layer_norm.out_quantizer.scale {qlayer.self_attn_layer_norm.out_quantizer.scale.shape}={qlayer.self_attn_layer_norm.out_quantizer.scale}")
        #print(f"qlayer.self_attn_layer_norm.out_quantizer.zero_point {qlayer.self_attn_layer_norm.out_quantizer.zero_point.shape}={qlayer.self_attn_layer_norm.out_quantizer.zero_point}")

        ic_maxmin_dict.clear()
        oc_maxmin_dict.clear()
        
        qlayer.self_attn.debug=1
        #qlayer.self_attn_layer_norm.debug=1
        
        layers[i] = qlayer.to("cpu")
        
        
        #print(f"layers[i].debug=={layers[i].debug}")
        #print("model:")
        #print(model)
        
        del layer
        torch.cuda.empty_cache()

        #print("Final qlayer:")
        ##print(qlayer)
        #
        #print("\n\n")
        #print(f"qlayer.self_attn.k_proj.weight: {qlayer.self_attn.k_proj.weight.shape}")
        #print(qlayer.self_attn.k_proj.weight)
        #print("\n\n")

        inps, outs = outs, inps
        print(
            lm._device,
            "memory_allocated",
            i,
            torch.cuda.memory_allocated(lm._device) / 1024 / 1024,
            "max memory_allocated",
            torch.cuda.max_memory_allocated(lm._device) / 1024**2,
        )

    del inps, outs
    model.config.use_cache = use_cache
    
    
    #print("After quantization, model:")
    #print(model)
    
    return model


if __name__ == "__main__":
    tensor = torch.rand([30])
    index, counts = tensor_calc_reorder_index(tensor, 2, 3)
    print(index, counts)
