import os
import sys

import random
import numpy as np
from models.opt import OPTClass
import torch
import time
from datautils import get_loaders
from lm_evaluation.lm_eval import tasks, evaluator
from quantize.opt_reorder_quantize import opt_reorder_quantize
import datetime
from models.int_opt_layer import QuantOPTAttention
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.opt_reorder_quantize import opt_reorder_quantize
from tqdm import tqdm

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    # "llama-7b",
    # "llama-13b",
    # "bloom-3b",
]

# tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq


@torch.no_grad()
def evaluate(lm, args):
    for name, m in lm.model.named_modules():
        if isinstance(m, (QuantOPTAttention,)):
            m.name = name
            # m.register_forward_hook(mem_test_hook)
    results = {}
    if args.multigpu:
        if "opt" in args.model:
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)

            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.model:
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
    else:
        if "opt" in args.model:
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.model:
            lm.model.model = lm.model.model.to(lm.device)

    
    #print(f"lm.model.model={lm.model.model}")
    if args.eval_ppl:
        for dataset in ["wikitext2", "ptb", "c4"]:
        #for dataset in ["wikitext2"]:
        #for dataset in ["ptb"]:
            # for dataset in ['c4']:
            if "opt" in args.model:
                cache_testloader = f"/home/liwj/project/RPTQ/RPTQ4LLM/tmp/{dataset}_testloader_opt_all.cache"
                if os.path.exists(cache_testloader):
                    testloader = torch.load(cache_testloader)
                    # print(f"load calibration from {cache_testloader}")
                else:
                    dataloader, testloader = get_loaders(
                        dataset,
                        seed=args.seed,
                        model=args.model,
                        seqlen=lm.seqlen,
                        cache_dir=args.cache_dir,
                    )
                    torch.save(testloader, cache_testloader)
            elif "llama" in args.model:
                cache_testloader = f"/home/liwj/project/RPTQ/RPTQ4LLM/tmp/{dataset}_testloader_llama_all.cache"
                if os.path.exists(cache_testloader):
                    testloader = torch.load(cache_testloader)
                    # print(f"load calibration from {cache_testloader}")
                else:
                    dataloader, testloader = get_loaders(
                        dataset,
                        seed=args.seed,
                        model=args.model,
                        seqlen=lm.seqlen,
                        cache_dir=args.cache_dir,
                    )
                    torch.save(testloader, cache_testloader)
            # print(dataset)
            if "c4" == dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            #print(f"testenc: {testenc.shape}")
            #print(testenc)

            nsamples = testenc.numel() // lm.seqlen   
            use_cache = lm.model.config.use_cache
            
            #print(f"lm.seqlen={lm.seqlen}, nsamples={nsamples}, use_cache={use_cache}")
            
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []

            print(f"range(nsamples)={range(nsamples)}, lm.seqlen={lm.seqlen}")
            for i in tqdm(range(nsamples)):
                #print(f"i={i}")
                #if(i==1):
                #    break
                
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(
                    lm.device
                )
                
                #batch = testenc[:, ((i+1) * lm.seqlen) : ((i + 2) * lm.seqlen)].to(
                #    lm.device
                #)
                
                #print(f"i={i}, batch ({batch.shape})={batch}")
                
                if "opt" in args.model:
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.model:
                    outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            print(dataset, ppl.item())
            lm.model.config.use_cache = use_cache
            # pprint(args.model)
            results[dataset] = ppl.item()
    
    
    '''
    print(f"args.tasks={args.tasks}")
    print(f"type(args.tasks)={type(args.tasks)}")    
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        pprint(results)
    '''
    return results


@torch.no_grad()
def calibrate_zero_task(lm, args, task_name):
    for name, m in lm.model.named_modules():
        if isinstance(m, (QuantOPTAttention,)):
            m.name = name
            # m.register_forward_hook(mem_test_hook)
    results = {}
    if args.multigpu:
        if "opt" in args.model:
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)

            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.model:
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
    else:
        if "opt" in args.model:
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.model:
            lm.model.model = lm.model.model.to(lm.device)

            
    if task_name != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=task_name,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        pprint(results)
    
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, choices=net_choices)
    parser.add_argument(
        "--cache_dir", default="./data", type=str, help="OPT model cache_dir"
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="mix",
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ema_minmax",
        choices=["minmax", "ema_minmax", "mse", "layer_mse"],
    )

    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output_path", default="./output")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--disable_w_quant", action="store_true")
    parser.add_argument("--disable_a_quant", action="store_true")
    
    # RPTQ for opt-6.7b
    #parser.add_argument("--R1_clusters", type=int, default=32)
    #parser.add_argument("--R2_clusters", type=int, default=1)
    #parser.add_argument("--R3_clusters", type=int, default=1)
    #parser.add_argument("--R4_clusters", type=int, default=32)
    #parser.add_argument("--R5_clusters", type=int, default=128)
    # RPTQ for opt-13b
    #parser.add_argument("--R1_clusters", type=int, default=40)
    #parser.add_argument("--R2_clusters", type=int, default=1)
    #parser.add_argument("--R3_clusters", type=int, default=1)
    #parser.add_argument("--R4_clusters", type=int, default=40)
    #parser.add_argument("--R5_clusters", type=int, default=160)
    # RPTQ test
    #parser.add_argument("--R1_clusters", type=int, default=32)
    #parser.add_argument("--R2_clusters", type=int, default=4)
    #parser.add_argument("--R3_clusters", type=int, default=4)
    #parser.add_argument("--R4_clusters", type=int, default=32)
    #parser.add_argument("--R5_clusters", type=int, default=32)
    
    parser.add_argument("--R1_clusters", type=int, default=1)
    parser.add_argument("--R2_clusters", type=int, default=1)
    parser.add_argument("--R3_clusters", type=int, default=1)
    parser.add_argument("--R4_clusters", type=int, default=1)
    parser.add_argument("--R5_clusters", type=int, default=1)
    
    parser.add_argument("--reorder", type=str, default="12345", help="like 12345 or 1")
    parser.add_argument(
        "--w_quantizer", type=str, default="gptq", choices=["gptq", "normal"]
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--a_dynamic", action="store_true")
    parser.add_argument("--eval_base_ppl", action="store_true")
    parser.add_argument("--act_dist_plot", action="store_true")
    parser.add_argument("--only_quant_kv", action="store_true")
    parser.add_argument(
        "--pack_weight",
        action="store_true",
        help="enable this to reduce memory consumption",
    )
    parser.add_argument(
        "--multigpu", action="store_true", help="at eval, map model to multiple gpus"
    )
    
    parser.add_argument("--topk_num", type=int, default=2)
    parser.add_argument("--topk_num_final_layer_norm", type=int, default=2)
    parser.add_argument("--topk_num_fc2", type=int, default=2)
    parser.add_argument("--topk_num_out_proj_head", type=int, default=2)
    parser.add_argument("--topk_num_q_proj_head", type=int, default=2)
    
    parser.add_argument("--group128", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=64)

    args = parser.parse_args()
    args.batch_size = 1  # BS=1 is used for zeroShot tasks!
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if "opt" in args.net:
        args.model = f"facebook/{args.net}"
        if not os.path.exists(f"{args.cache_dir}/{args.net.split('-')[0]}/"):
            os.makedirs(f"{args.cache_dir}/{args.net.split('-')[0]}/")
        args.cache_dir = (
            f"{args.cache_dir}/{args.net.split('-')[0]}/{args.net.split('-')[1]}"
        )
        print(args.cache_dir)
        cache_file = f"{args.cache_dir}/torch_model.pth"
        if os.path.exists(cache_file):
            lm = torch.load(cache_file)
            
            sum_activation=torch.zeros([40,5120]).to('cpu')
            lm.model.model.decoder.register_buffer('sum_activation', sum_activation)
            #print(f"self.sum_activation={lm.model.model.decoder.sum_activation}")
            print(f"lm.model={lm.model}")
            
        else:
            lm = OPTClass(args)
            torch.save(lm, cache_file)
        lm.model.eval()
    else:
        raise NotImplementedError



    
    #num_layer = 0
    #embed_dim = 0
    #for layer in lm.model.model.decoder.layers:
    #    num_layer = num_layer + 1
    #    embed_dim = layer.embed_dim
    #
    #weight_Q = torch.zeros(num_layer,embed_dim)
    #weight_K = torch.zeros(num_layer,embed_dim)
    #weight_V = torch.zeros(num_layer,embed_dim)
    #weight_out_proj = torch.zeros(num_layer,embed_dim)
    #weight_fc1 = torch.zeros(num_layer,embed_dim)
    #weight_fc2 = torch.zeros(num_layer,4*embed_dim)
    #
    #idx=0
    #for layer in lm.model.model.decoder.layers:
    #    
    #    #print(f"idx={idx}: layer={layer}")
    #    layer.idx=idx
    #    #sum_fc1=torch.zeros([1,5120]).cpu()
    #    #sum_fc2=torch.zeros([1,20480]).cpu()
    #    #layer.register_buffer('sum_fc1', sum_fc1)
    #    #layer.register_buffer('sum_fc2', sum_fc2)
    #    
    #    
    #    #print(f"idx={idx}")
    #    for name, module in layer.named_modules():
    #        #print(f"name={name},module={module}")
    #        #if(name=='self_attn'):
    #        #    module.idx = idx
    #        #    sum_q=torch.zeros([1,5120]).cpu()
    #        #    sum_k=torch.zeros([1,5120]).cpu()
    #        #    sum_v=torch.zeros([1,5120]).cpu()
    #        #    sum_out=torch.zeros([1,5120]).cpu()
    #        #    #print(f"sum_q.shape={sum_q.shape},sum_k.shape={sum_k.shape}")
    #        #    module.register_buffer('sum_q', sum_q)
    #        #    module.register_buffer('sum_k', sum_k)
    #        #    module.register_buffer('sum_v', sum_v)
    #        #    module.register_buffer('sum_out', sum_out)
    #        #    #print(f"module.sum_q={module.sum_q}")
    #        #    #print(f"module.idx={module.idx}")
    #            
    #        if(name=='self_attn.k_proj' or name=='self_attn.v_proj' or name=='self_attn.q_proj' or name=='self_attn.out_proj' or name=='fc1' or name=='fc2'):
    #            #print(f"module.weight.shape={module.weight.shape}")
    #            tmp_weight = torch.sum(torch.abs(module.weight),dim=0) / module.weight.shape[0]
    #            
    #            tmp_weight_sorted, tmp_weight_idx = torch.sort(tmp_weight,descending=True)
    #            if(name=='self_attn.q_proj'):
    #                #print(f"idx{idx}, tmp_weight_sorted[10]={tmp_weight_sorted[0:10].data.numpy()}")
    #                if(idx!=-1):
    #                    #print(module.weight[:,501:505])
    #                    #print(f"larger than 0.008: {torch.count_nonzero((torch.abs(module.weight)>0.008).long()) / module.weight.shape[0] / module.weight.shape[1]}")
    #                    max_weight_value, max_weight_idx = torch.max(torch.abs(module.weight),dim=0)
    #                    print(f"max_weight_value={max_weight_value}")
    #                    print(f"max_weight_idx={max_weight_idx}")
    #            
    #            if(name=='self_attn.q_proj'):
    #                weight_Q[idx] = tmp_weight
    #            if(name=='self_attn.k_proj'):
    #                weight_K[idx] = tmp_weight
    #            if(name=='self_attn.v_proj'):
    #                weight_V[idx] = tmp_weight
    #            if(name=='self_attn.out_proj'):
    #                weight_out_proj[idx] = tmp_weight
    #            if(name=='fc1'):
    #                weight_fc1[idx] = tmp_weight
    #            if(name=='fc2'):
    #                weight_fc2[idx] = tmp_weight
    #            
    #            
    #            #if(name=='self_attn.k_proj'):
    #            #    filename="weight_k_proj.txt"
    #            #if(name=='self_attn.v_proj'):
    #            #    filename="weight_v_proj.txt"
    #            #if(name=='self_attn.q_proj'):
    #            #    filename="weight_q_proj.txt"
    #            #if(name=='self_attn.out_proj'):
    #            #    filename="weight_out_proj.txt"
    #            #if(name=='fc1'):
    #            #    filename="weight_fc1.txt"
    #            #if(name=='fc2'):
    #            #    filename="weight_fc2.txt"
    #            #fp=open(filename,'a')
    #            #for iii in range(module.weight.shape[-1]):
    #            #    fp.write(str(tmp_weight[iii].item())+' ')
    #            #fp.write('\n')
    #            #fp.close()
    #    idx=idx+1
    #
    #average_Q = torch.sum(weight_Q,dim=1) / weight_Q.shape[1]
    #average_K = torch.sum(weight_K,dim=1) / weight_K.shape[1]
    #average_V = torch.sum(weight_V,dim=1) / weight_V.shape[1]
    #average_out_proj = torch.sum(weight_out_proj,dim=1) / weight_out_proj.shape[1]
    #average_fc1 = torch.sum(weight_fc1,dim=1) / weight_fc1.shape[1]
    #average_fc2 = torch.sum(weight_fc2,dim=1) / weight_fc2.shape[1]
    #for idx in range(num_layer): 
    #    num_Q = torch.count_nonzero((weight_Q[idx] > average_Q[idx]).long())
    #    num_K = torch.count_nonzero((weight_K[idx] > average_K[idx]).long())
    #    num_V = torch.count_nonzero((weight_V[idx] > average_V[idx]).long())
    #    num_out_proj = torch.count_nonzero((weight_out_proj[idx] > average_out_proj[idx]).long())
    #    num_fc1 = torch.count_nonzero((weight_fc1[idx] > average_fc1[idx]).long())
    #    num_fc2 = torch.count_nonzero((weight_fc2[idx] > average_fc2[idx]).long())
    #    print(f"layer{idx}: {num_Q.item()}, {num_K.item()}, {num_V.item()}, {num_out_proj.item()}, {num_fc1.item()}, {num_fc2.item()}")
    #    print(f"average: {average_Q[idx].item()}, {average_K[idx].item()}, {average_V[idx].item()}, {average_out_proj[idx].item()}, {average_fc1[idx].item()}, {average_fc2[idx].item()}")
        
        

    
    layers = lm.model.model.decoder.layers
    for i in range(len(layers)):
        layer = layers[i]
        layer.num_inference = 0
        layer.calibrate_ZS_task_en = 0
        layer.calibrate_ZS_task_num = 0
        for name, module in layer.named_modules():
            if(name=='self_attn'):
                module.num_inference = 0
                module.calibrate_ZS_task_en = 0
                module.calibrate_ZS_task_num = 0


    '''
    print("=== calibration for zero-shot tasks ===")
    layers = lm.model.model.decoder.layers
    #print(f"layers={layers}")
    num_layer = 0
    embed_dim = 0
    for i in range(len(layers)):
        #print(f"layers[i]={layers[i]}")
        layer = layers[i]
        num_layer = num_layer + 1
        embed_dim = layer.embed_dim
        
        fc1_X = torch.zeros(2,layer.embed_dim)
        fc2_X = torch.zeros(2,4*layer.embed_dim)
        layer.num_inference = 0
        layer.calibrate_ZS_task_en = 0
        layer.calibrate_ZS_task_num = 0
        layer.register_buffer('fc1_X', fc1_X)
        layer.register_buffer('fc2_X', fc2_X)
        for name, module in layer.named_modules():
            #print(f"name={name}")
            if(name=='self_attn'):
                #print("!!!")
                first_X = torch.zeros(2,module.embed_dim)
                Q = torch.zeros(2,module.embed_dim)
                out_proj_X = torch.zeros(2,module.embed_dim)
                module.num_inference = 0
                module.calibrate_ZS_task_en = 0
                module.calibrate_ZS_task_num = 0
                module.register_buffer('first_X', first_X)
                module.register_buffer('Q', Q)
                module.register_buffer('out_proj_X', out_proj_X)
                
    # lambada_openai
    #for i in range(len(layers)):
    #    layer = layers[i]
    #    layer.num_inference = 0
    #    layer.calibrate_ZS_task_en = 1
    #    layer.calibrate_ZS_task_num = 512
    #    for name, module in layer.named_modules():
    #        if(name=='self_attn'):
    #            module.num_inference = 0
    #            module.calibrate_ZS_task_en = 1
    #            module.calibrate_ZS_task_num = 512
    #results = calibrate_zero_task(lm, args, 'lambada_openai')
    #
    #fc1_lambada_openai = torch.zeros(num_layer,embed_dim)
    #fc2_lambada_openai = torch.zeros(num_layer,4*embed_dim)
    #first_X_lambada_openai = torch.zeros(num_layer,embed_dim)
    #Q_lambada_openai = torch.zeros(num_layer,embed_dim)
    #out_proj_X_lambada_openai = torch.zeros(num_layer,embed_dim)
    #
    #for i in range(len(layers)):
    #    layer = layers[i]
    #    layer.calibrate_ZS_task_en = 0
    #    
    #    fc1_lambada_openai[i,:] = (torch.abs(layer.fc1_X[0])+torch.abs(layer.fc1_X[1]))/2
    #    fc2_lambada_openai[i,:] = (torch.abs(layer.fc2_X[0])+torch.abs(layer.fc2_X[1]))/2
    #    
    #    for name, module in layer.named_modules():
    #        if(name=='self_attn'):
    #            layer.calibrate_ZS_task_en = 0
    #            
    #            first_X_lambada_openai[i,:] = (torch.abs(module.first_X[0])+torch.abs(module.first_X[1]))/2
    #            Q_lambada_openai[i,:] = (torch.abs(module.Q[0])+torch.abs(module.Q[1]))/2
    #            out_proj_X_lambada_openai[i,:] = (torch.abs(module.out_proj_X[0])+torch.abs(module.out_proj_X[1]))/2
    #print("Writing files about lambada_openai ...")            
    #fp=open('first_X_lambada_openai.txt','a')
    #for i in range(first_X_lambada_openai.shape[0]):
    #    for j in range(first_X_lambada_openai.shape[1]):
    #        fp.write(str(first_X_lambada_openai[i][j].item())+' ')
    #    fp.write('\n')
    #fp.close()
    #fp=open('Q_lambada_openai.txt','a')
    #for i in range(Q_lambada_openai.shape[0]):
    #    for j in range(Q_lambada_openai.shape[1]):
    #        fp.write(str(Q_lambada_openai[i][j].item())+' ')
    #    fp.write('\n')
    #fp.close()
    #fp=open('out_proj_X_lambada_openai.txt','a')
    #for i in range(out_proj_X_lambada_openai.shape[0]):
    #    for j in range(out_proj_X_lambada_openai.shape[1]):
    #        fp.write(str(out_proj_X_lambada_openai[i][j].item())+' ')
    #    fp.write('\n')
    #fp.close()
    #fp=open('fc1_lambada_openai.txt','a')
    #for i in range(fc1_lambada_openai.shape[0]):
    #    for j in range(fc1_lambada_openai.shape[1]):
    #        fp.write(str(fc1_lambada_openai[i][j].item())+' ')
    #    fp.write('\n')
    #fp.close()
    #fp=open('fc2_lambada_openai.txt','a')
    #for i in range(fc2_lambada_openai.shape[0]):
    #    for j in range(fc2_lambada_openai.shape[1]):
    #        fp.write(str(fc2_lambada_openai[i][j].item())+' ')
    #    fp.write('\n')
    #fp.close()
    
    # openbookqa
    for i in range(len(layers)):
        layer = layers[i]
        layer.num_inference = 0
        layer.calibrate_ZS_task_en = 1
        layer.calibrate_ZS_task_num = 512
        for name, module in layer.named_modules():
            if(name=='self_attn'):
                module.num_inference = 0
                module.calibrate_ZS_task_en = 1
                module.calibrate_ZS_task_num = 512
    results = calibrate_zero_task(lm, args, 'openbookqa')
    
    fc1_openbookqa = torch.zeros(num_layer,embed_dim)
    fc2_openbookqa = torch.zeros(num_layer,4*embed_dim)
    first_X_openbookqa = torch.zeros(num_layer,embed_dim)
    Q_openbookqa = torch.zeros(num_layer,embed_dim)
    out_proj_X_openbookqa = torch.zeros(num_layer,embed_dim)
    
    for i in range(len(layers)):
        layer = layers[i]
        layer.calibrate_ZS_task_en = 0
        
        fc1_openbookqa[i,:] = (torch.abs(layer.fc1_X[0])+torch.abs(layer.fc1_X[1]))/2
        fc2_openbookqa[i,:] = (torch.abs(layer.fc2_X[0])+torch.abs(layer.fc2_X[1]))/2
        
        for name, module in layer.named_modules():
            if(name=='self_attn'):
                layer.calibrate_ZS_task_en = 0
                
                first_X_openbookqa[i,:] = (torch.abs(module.first_X[0])+torch.abs(module.first_X[1]))/2
                Q_openbookqa[i,:] = (torch.abs(module.Q[0])+torch.abs(module.Q[1]))/2
                out_proj_X_openbookqa[i,:] = (torch.abs(module.out_proj_X[0])+torch.abs(module.out_proj_X[1]))/2
    print("Writing files about openbookqa ...")            
    fp=open('first_X_openbookqa.txt','a')
    for i in range(first_X_openbookqa.shape[0]):
        for j in range(first_X_openbookqa.shape[1]):
            fp.write(str(first_X_openbookqa[i][j].item())+' ')
        fp.write('\n')
    fp.close()
    fp=open('Q_openbookqa.txt','a')
    for i in range(Q_openbookqa.shape[0]):
        for j in range(Q_openbookqa.shape[1]):
            fp.write(str(Q_openbookqa[i][j].item())+' ')
        fp.write('\n')
    fp.close()
    fp=open('out_proj_X_openbookqa.txt','a')
    for i in range(out_proj_X_openbookqa.shape[0]):
        for j in range(out_proj_X_openbookqa.shape[1]):
            fp.write(str(out_proj_X_openbookqa[i][j].item())+' ')
        fp.write('\n')
    fp.close()
    fp=open('fc1_openbookqa.txt','a')
    for i in range(fc1_openbookqa.shape[0]):
        for j in range(fc1_openbookqa.shape[1]):
            fp.write(str(fc1_openbookqa[i][j].item())+' ')
        fp.write('\n')
    fp.close()
    fp=open('fc2_openbookqa.txt','a')
    for i in range(fc2_openbookqa.shape[0]):
        for j in range(fc2_openbookqa.shape[1]):
            fp.write(str(fc2_openbookqa[i][j].item())+' ')
        fp.write('\n')
    fp.close()
    '''    


    
    print("=== start quantization ===")
    if args.load:
        print("Loading checkpoint from {}...".format(args.load))
        lm.model.load_state_dict(torch.load(args.load))

    tick = time.time()

    if "opt" in args.model:
        cache_dataloader = (
            f"/home/liwj/project/RPTQ/RPTQ4LLM/tmp/dataloader_opt_{args.calib_dataset}_{args.nsamples}.cache"
        )
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            print(f"load calibration from {cache_dataloader}")
        else:
            dataloader, testloader = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
                cache_dir=args.cache_dir,
            )
            torch.save(dataloader, cache_dataloader)
        lm.model.eval()
    else:
        raise NotImplementedError()
    
    ################## lwj ##################
    #torch.cuda.set_device(2)
    #device_count = torch.cuda.device_count()
    #print(f"Number of available GPUs: {device_count}")
    #for i in range(device_count):
    #    device_name = torch.cuda.get_device_name(i)
    #    print(f"GPU device {i}: {device_name}")
    #
    #print(f"lm.device={lm.device}")
    ################## lwj ##################
    
    
    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": False,
        "metric": "minmax",
    }
    args.act_quant_params = {
        "n_bits": 16 if args.only_quant_kv else args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.q_quant_params = {
        "n_bits": 16 if args.only_quant_kv else args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.layer_norm_out_quant_params = {
        "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.p_quant_params = {
        "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
        "metric": "fix0to1",
    }
    n_clusters = {
        "R1": args.R1_clusters,
        "R2": args.R2_clusters,
        "R3": args.R3_clusters,
        "R4": args.R4_clusters,
        "R5": args.R5_clusters,
    }
    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        print(f"set quantization in gpu {gpu_id}")
    if "opt" in args.model:
    
        #print("Model before quantization:")
        #print(lm.model.model.decoder.layers)
        
        opt_reorder_quantize(
            lm,
            args,
            dataloader,
            n_clusters,
            args.reorder,
        )

        #print("Model after quantization:")
        #print(lm.model.model.decoder.layers)

        for layer in lm.model.model.decoder.layers:
            if hasattr(layer, "set_quant_state"):
                layer.set_quant_state(
                    not args.disable_w_quant, not args.disable_a_quant
                )

    print(time.time() - tick)
    

    #print("After quantization")
    #idx=0
    #for layer in lm.model.model.decoder.layers:
    #    #print(f"idx={idx},layer={layer}")
    #    #print(f"idx={idx}")
    #    for name, module in layer.named_modules():
    #        #print(f"name={name}")
    #    
    #        if(name=='self_attn.k_proj' or name=='self_attn.v_proj' or name=='self_attn.q_proj' or name=='self_attn.out_proj' or name=='fc1' or name=='fc2'):
    #            #print(f"module.weight.shape={module.weight.shape}")
    #            tmp_weight = torch.sum(torch.abs(module.weight),dim=0) / module.weight.shape[0]
    #            if(name=='self_attn.k_proj'):
    #                filename="quant_weight_k_proj.txt"
    #            if(name=='self_attn.v_proj'):
    #                filename="quant_weight_v_proj.txt"
    #            if(name=='self_attn.q_proj'):
    #                filename="quant_weight_q_proj.txt"
    #            if(name=='self_attn.out_proj'):
    #                filename="quant_weight_out_proj.txt"
    #            if(name=='fc1'):
    #                filename="quant_weight_fc1.txt"
    #            if(name=='fc2'):
    #                filename="quant_weight_fc2.txt"
    #            fp=open(filename,'a')
    #            for iii in range(module.weight.shape[-1]):
    #                fp.write(str(tmp_weight[iii].item())+' ')
    #            fp.write('\n')
    #            fp.close()
    #        
    #    idx=idx+1
    
    
    #for layer in lm.model.model.decoder.layers:
    #    layer.self_attn.k_proj.use_act_quant=False
    #    layer.self_attn.k_proj.disable_input_quant=True
    #    layer.self_attn.v_proj.use_act_quant=False
    #    layer.self_attn.v_proj.disable_input_quant=True
    #    layer.self_attn.q_proj.use_act_quant=False
    #    layer.self_attn.q_proj.disable_input_quant=True
    #    layer.self_attn.out_proj.use_act_quant=False
    #    layer.self_attn.out_proj.disable_input_quant=True
    #    layer.self_attn.qkt_matmul.use_act_quant=False
    #    layer.self_attn.qkt_matmul.disable_input_quant=True
    #    layer.self_attn.qkt_matmul.dis_x1_quant=1
    #    layer.self_attn.qkt_matmul.dis_x2_quant=1
    #    
    #    layer.self_attn.pv_matmul.use_act_quant=False
    #    layer.self_attn.pv_matmul.disable_input_quant=True
    #    layer.self_attn.pv_matmul.dis_x1_quant=1
    #    layer.self_attn.pv_matmul.dis_x2_quant=1
    #    
    #    
    #    layer.self_attn_layer_norm.use_act_quant=False
    #    layer.fc1.use_act_quant=False
    #    layer.fc1.disable_input_quant=True
    #    layer.fc2.use_act_quant=False
    #    layer.fc2.disable_input_quant=True
    #    layer.final_layer_norm.use_act_quant=False
        
        
    
    
    results = evaluate(lm, args)
    
    #max_vals, _ = torch.max(lm.model.model.decoder.sum_activation, dim=1)
    #index=torch.argmax(lm.model.model.decoder.sum_activation, dim=1)
    #print(f"max_vals.unsqueeze(1)={max_vals.unsqueeze(1)}")
    #print(f"index={index}")
    #lm.model.model.decoder.sum_activation = lm.model.model.decoder.sum_activation / max_vals.unsqueeze(1)
    #
    #
    #
    #print(f"lm.model.model.decoder.sum_activation.shape={lm.model.model.decoder.sum_activation.shape}")
    #print(f"lm.model.model.decoder.sum_activation[:,900:905]={lm.model.model.decoder.sum_activation[:,900:905]}")
    
    #fp=open("sum_activation.txt","w")
    #for ii in range(40):
    #    print(f"ii={ii}")
    #    for jj in range(lm.model.model.decoder.sum_activation.shape[-1]):
    #        fp.write(str(lm.model.model.decoder.sum_activation[ii][jj].item())+' ')
    #    fp.write('\n')
    #fp.close()
    
    #for ii in range(40):
    #    fp=open("sum_activation_"+str(ii)+".txt","w")
    #    for jj in range(lm.model.model.decoder.sum_activation.shape[-1]):
    #        fp.write(str(lm.model.model.decoder.sum_activation[ii][jj].item())+'\n')
    #    fp.close()
    
    
    #import matplotlib.pyplot as plt
    #x, y = np.meshgrid(np.arange(lm.model.model.decoder.sum_activation.shape[1]), np.arange(lm.model.model.decoder.sum_activation.shape[0]))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(x, y, lm.model.model.decoder.sum_activation, cmap=plt.get_cmap('rainbow'))
    #fig.colorbar(surf, shrink=0.5, aspect=5,label='Z Value')
    #
    #ax.set_xlabel('channel')
    #ax.set_ylabel('length')
    
    #plt.bar(range(len(lm.model.model.decoder.sum_activation[2])), lm.model.model.decoder.sum_activation[2])
    
    #plt.show()
    #fig.savefig(str(idx)+".pdf")
    
    
    
    
    
    #idx=0
    #for layer in lm.model.model.decoder.layers:
    #    #print(f"idx={idx}: layer={layer}")
    #    print(f"idx={idx}")
    #    
    #    
    #    #layer.sum_fc1=layer.sum_fc1/ 140 / lm.seqlen
    #    #layer.sum_fc2=layer.sum_fc2/ 140 / lm.seqlen
    #    #
    #    #fp=open("sum_fc1.txt","a")
    #    #for jj in range(layer.sum_fc1.shape[-1]):
    #    #    fp.write(str(layer.sum_fc1[0][jj].item())+' ')
    #    #fp.write('\n')
    #    #fp.close()
    #    #
    #    #fp=open("sum_fc2.txt","a")
    #    #for jj in range(layer.sum_fc2.shape[-1]):
    #    #    fp.write(str(layer.sum_fc2[0][jj].item())+' ')
    #    #fp.write('\n')
    #    #fp.close()
    #    
    #    for name, module in layer.named_modules():
    #        if(name=='self_attn'):
    #            #print(f"name={name},module={module}")
    #            
    #            #print(f"module.sum_q.shape={module.sum_q.shape}:")
    #            #print(module.sum_q)
    #            #print(f"module.sum_k.shape={module.sum_k.shape}:")
    #            #print(module.sum_k)
    #            
    #            #max_vals, index = torch.max(module.sum_q, dim=1)
    #            ##print(f"max_vals={max_vals},index={index}")
    #            #module.sum_q=module.sum_q/max_vals
    #            #
    #            #max_vals, index = torch.max(module.sum_k, dim=1)
    #            ##print(f"max_vals={max_vals},index={index}")
    #            #module.sum_k=module.sum_k/max_vals
    #            
    #            # wikitext2: 140
    #            #module.sum_q = module.sum_q / 140 / lm.seqlen
    #            #module.sum_k = module.sum_k / 140 / lm.seqlen
    #            #module.sum_v = module.sum_v / 140 / lm.seqlen
    #            module.sum_out = module.sum_out / 140 / lm.seqlen
    #            
    #            #fp=open("sum_q.txt","a")
    #            #for jj in range(module.sum_q.shape[-1]):
    #            #    fp.write(str(module.sum_q[0][jj].item())+' ')
    #            #fp.write('\n')
    #            #fp.close()
    #            
    #            #fp=open("sum_k.txt","a")
    #            #for jj in range(module.sum_k.shape[-1]):
    #            #    fp.write(str(module.sum_k[0][jj].item())+' ')
    #            #fp.write('\n')
    #            #fp.close()
    #            
    #            #fp=open("sum_v.txt","a")
    #            #for jj in range(module.sum_v.shape[-1]):
    #            #    fp.write(str(module.sum_v[0][jj].item())+' ')
    #            #fp.write('\n')
    #            #fp.close()
    #            
    #            fp=open("sum_out.txt","a")
    #            for jj in range(module.sum_out.shape[-1]):
    #                fp.write(str(module.sum_out[0][jj].item())+' ')
    #            fp.write('\n')
    #            fp.close()
    #            
    #    idx=idx+1
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(
        f"{args.output_path}/{args.net}.txt",
        "a+",
    ) as f:
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"{' '.join(sys.argv)} {formatted_time} \n {args} \n w{args.wbits}a{args.abits} {results}\n\n"
        )
    

if __name__ == "__main__":
    print(sys.argv)
    main()
