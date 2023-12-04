# wenjie_github_TCAS1

This is the code for our proposed quantization scheme, which is based on the open-source code of RPTQ. You can find it on https://github.com/hahnyuan/RPTQ4LLM/tree/master. Besides, please see the required python packages and the base commands in the repository of RPTQ. 

The main changes we made lie in the following files:
1. main.py; 2. quantize/opt_reorder_quantize.py; 3. quantize/quantizer.py.

# Reproduce the results of our proposed quantization scheme
To reproduce the perplexity results for our proposed quantization scheme, please first choose corresponding arguments in the main.py:

// for opt-1.3b, opt-6.7b and opt-13b  
// grouping will be performed by the code corresponding to lines 465 to 644 of quantize/opt_reorder_quantize.py  
```
parser.add_argument("--R1_clusters", type=int, default=1)  
parser.add_argument("--R2_clusters", type=int, default=1)  
parser.add_argument("--R3_clusters", type=int, default=1)  
parser.add_argument("--R4_clusters", type=int, default=1)  
parser.add_argument("--R5_clusters", type=int, default=1)
```

// for opt-1.3b, opt-6.7b and opt-13b: in the top-k algorithm, K=8  
```
parser.add_argument("--topk_num", type=int, default=8)  
parser.add_argument("--topk_num_final_layer_norm", type=int, default=8)  
parser.add_argument("--topk_num_fc2", type=int, default=8)  
parser.add_argument("--topk_num_out_proj_head", type=int, default=8)  
parser.add_argument("--topk_num_q_proj_head", type=int, default=8)
```

// for opt-1.3b, opt-6.7b and opt-13b: enable grouping  
```
parser.add_argument("--group128", type=int, default=1)
```

// for opt-1.3b: group size  
```
parser.add_argument("--group_size", type=int, default=64)
```

// for opt-6.7b and opt-13b: group size  
```
parser.add_argument("--group_size", type=int, default=128)
```

# Reproduce the results of RPTQ
To reproduce the perplexity results for RPTQ, please first choose corresponding arguments in the main.py:

// for opt-1.3b and opt-6.7b  
```
parser.add_argument("--R1_clusters", type=int, default=32)  
parser.add_argument("--R2_clusters", type=int, default=1)  
parser.add_argument("--R3_clusters", type=int, default=1)  
parser.add_argument("--R4_clusters", type=int, default=32)  
parser.add_argument("--R5_clusters", type=int, default=128)  
```

// for opt-13b  
```
parser.add_argument("--R1_clusters", type=int, default=40)  
parser.add_argument("--R2_clusters", type=int, default=1)  
parser.add_argument("--R3_clusters", type=int, default=1)  
parser.add_argument("--R4_clusters", type=int, default=40)  
parser.add_argument("--R5_clusters", type=int, default=160)
```

It guarantees that the group numbers in RPTQ are equal to those used in our quantization scheme.

Then, please comment out the code realated to our quantization scheme in quantize/opt_reorder_quantize.py: comment out the lines from 465 to 644.

Also, remove the comment for lines 651 to 655 (R1), lines 770 to 774 (R4), and lines 803 to 807 (R5).

# Complete .zip file with datasets
If you use the code given in this repository, the datasets and models will be automatically downloaded. For your convenience, we also upload the .zip file not only including the code but also the required datasets. Note We did not provide the model files as they are too large in size. Please run the code and them will be automatically downloaded.

Here is the link:  
https://drive.google.com/file/d/1iHQD35v4WLijUksuTZAwYDOSAsIYt-aQ/view?usp=drive_link
