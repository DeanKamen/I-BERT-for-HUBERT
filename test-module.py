import contextlib
from itertools import chain
import logging
import sys
import time
from typing import Any, Dict, List

import torch
import numpy as np
import os
from fairseq.modules import TransformerSentenceEncoderLayer
from fairseq.models.roberta import RobertaModel
from fairseq.quantization.utils.quant_modules import *

# INSTRUCTIONS
# Run this script in the I-BERT directory when on branch "ibert" rather than "ibert-base" as this needs access to the
# QUANTIZED models, rather than the normal bert model. To convert to bert model, see comments on the "from_pretrained" function.
# before running, change the paths to be compatible with your system, as they are not portable

# USAGE
# This script instantiates a module from roberta so you can test it. 

# the following paths aren't fexible and have to be changed when userspace changes or models are requantized
RTEPATH = "outputs/symmetric/RTE-base/wd0.1_ad0.1_d0.1_lr1e-06/0601-140215_ckpt/"
RTEMODEL = "/home/messn036/I-BERT/RTE-bin" 
SST2PATH = "outputs/symmetric/SST-2-base/wd0.1_ad0.1_d0.1_lr1e-06/0601-142002_ckpt/"
SST2MODEL = "/home/messn036/I-BERT/SST-2-bin"
MNLIPATH = "outputs/symmetric/MNLI-base/wd0.1_ad0.1_d0.1_lr1e-06/0601-164842_ckpt/"
MNLIMODEL = "/home/messn036/I-BERT/MNLI-bin"
QNLIPATH = "outputs/symmetric/QNLI-base/wd0.1_ad0.1_d0.1_lr1e-06/0602-042005_ckpt/"
QNLIMODEL = "/home/messn036/I-BERT/QNLI-bin"
CoLAPATH = "outputs/symmetric/CoLA-base/wd0.1_ad0.1_d0.1_lr1e-06/0603-032352_ckpt/"
CoLAMODEL = "/home/messn036/I-BERT/CoLA-bin"
QQPPATH = "outputs/symmetric/QQP-base/wd0.1_ad0.1_d0.1_lr1e-06/0603-040136_ckpt/"
QQPMODEL = "/home/messn036/I-BERT/QQP-bin"
MRPCPATH = "outputs/symmetric/MRPC-base/wd0.1_ad0.1_d0.1_lr1e-06/0603-123814_ckpt/"
MRPCMODEL = "/home/messn036/I-BERT/MRPC-bin"
STSBPATH = "outputs/symmetric/STS-B-base/wd0.1_ad0.1_d0.1_lr1e-06/0603-125613_ckpt/"
STSBMODEL = "/home/messn036/I-BERT/STS-B-bin"

#listify the GLUE tests
CKPT_PATHS = (RTEPATH, SST2PATH, MNLIPATH, QNLIPATH, CoLAPATH, QQPPATH, MRPCPATH, STSBPATH)
MODELS = (RTEMODEL, SST2MODEL, MNLIMODEL, QNLIMODEL, CoLAMODEL, QQPMODEL, MRPCMODEL, STSBMODEL)
TESTNAMES = ("RTE", "SST-2", "MNLI", "QNLI", "CoLA", "QQP", "MRPC", "STS-B")
 
def get_modulename(roberta, modulename): #gets Transformer Sentence Encoder Layer 0
    for name,module in roberta.named_modules():
        #print(name) #uncomment to see all your options.
        if(name == modulename): #this is the first TransformerSentenceEncoderLayer
        #note, to add more, print name to a file and add the desired layer as criterion here
            return module
    return None    

def main():
    roberta = RobertaModel.from_pretrained(MNLIPATH, #RTEPATH when quantized, models/roberta.base/ when not
                                        checkpoint_file = "checkpoint_best.pt", #checkpoint_best.pt when quantized, model.pt when not
                                        data_name_or_path = MNLIMODEL, 
                                        quant_mode = "symmetric", #symmetric when quantized, "none" when not
                                        force_dequant = "layernorm", #"layernorm" when quantized, "none" when not
                                        ) #data_name_or_path is need if using input data (inference)

    #The following block of code runs roberta through a prediction task, activating all relevant submodules
    #go into quant_modules.py and add some print statement to see output on a known working toy example
    
    roberta.register_classification_head('new_task', num_classes=3)# since I want to just execute prediction, I dont really care about the task.
    roberta.eval()
    roberta=roberta.cuda()
    #tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    #logprobs = roberta.predict('new_task', tokens) #new task doesnt measure anything, just here for ease of access
    #print(logprobs)

    #the next block loads layer 0 and feeds it realistic data from the encoding above. 
    #the encoding/prediction from above should be commented out to keep print statments from layer 0 only.

    #get_modulename(roberta, "dne") # prints out models if the print statement is uncommented in get_modulename
    
    print("START MODULE BREAKDOWN")
    testModule = get_modulename(roberta, "model.encoder.sentence_encoder.layers.0") #now this is the module we want information on
    testModule = testModule.cuda() #required, otherwise we are default loaded onto cpu

    a = np.load("export/layer0input.npz")
    layer0inputs = torch.from_numpy(a['layer0'])
    layer0inputs = layer0inputs.cuda()
    testModule(layer0inputs, None)

    return

    data = [[1, 2, 3]]
    test1 = torch.tensor(data).cuda()
    #data = [[0.4, 0.6], [2, 300], [3, 4000]]
    test2 = torch.tensor(data).cuda()
    #data = [[1, 1, 10], [1, 0, 30], [1, 2, 4], [3, 4, 6]]
    test3 = torch.tensor(data).cuda()
    #TESTDATA = (test1, test2, test3)
    TESTDATA = (test1,)
    #TESTDATA = () #skip this first test
    scalingfactor = 0.001

    i = 1
    for el in TESTDATA:
        print("INPUT ", i, " scaling factor: ", scalingfactor)
        print(el, "\n")
        result, ignore = testModule(el, torch.tensor(scalingfactor).cuda())
        print("OUTPUT ", i)
        print(result, "\n")
        i += 1

    testQuantAct = get_modulename(roberta, "model.encoder.sentence_encoder.layers.0.self_attn.softmax.act")
    testQuantAct = testQuantAct.cuda()
    data = [[5.0533e14, 1.1046e15, 2.9980e15]]
    x1 = torch.tensor(data).cuda()
    data = 3.3355e-16
    sf1 = torch.tensor(data).cuda()
    quant_act_int, act_scaling_factor = testQuantAct(x1, sf1)
    print("quant_act_int", quant_act_int, "act_scaling_factor", act_scaling_factor, "\n")
    
  
if __name__=="__main__":
    main()
