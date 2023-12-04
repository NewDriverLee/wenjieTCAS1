#!/bin/bash


python main.py opt-1.3b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 512
python main.py opt-1.3b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 256
python main.py opt-1.3b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 128
#python main.py opt-1.3b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 64
python main.py opt-1.3b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 32
python main.py opt-1.3b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 16
python main.py opt-1.3b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 8

#python main.py opt-1.3b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 512
##python main.py opt-1.3b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 256
#python main.py opt-1.3b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 128
##python main.py opt-1.3b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 64
#python main.py opt-1.3b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 32
#python main.py opt-1.3b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 16
#python main.py opt-1.3b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 8






python main.py opt-6.7b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 512
python main.py opt-6.7b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 256
#python main.py opt-6.7b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 128
python main.py opt-6.7b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 64
python main.py opt-6.7b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 32
python main.py opt-6.7b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 16
python main.py opt-6.7b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 8

#python main.py opt-6.7b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 512
##python main.py opt-6.7b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 256
##python main.py opt-6.7b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 128
#python main.py opt-6.7b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 64
#python main.py opt-6.7b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 32
#python main.py opt-6.7b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 16
#python main.py opt-6.7b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 8



python main.py opt-13b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 512
python main.py opt-13b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 256
#python main.py opt-13b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 128
python main.py opt-13b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 64
python main.py opt-13b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 32
python main.py opt-13b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 16
python main.py opt-13b --wbits 4 --abits 8 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 8

#python main.py opt-13b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 512
#python main.py opt-13b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 256
##python main.py opt-13b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 128
#python main.py opt-13b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 64
#python main.py opt-13b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 32
#python main.py opt-13b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 16
#python main.py opt-13b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu --group128 1 --group_size 8