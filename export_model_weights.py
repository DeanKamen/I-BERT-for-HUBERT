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

# INSTRUCTIONS
# Run this script in the I-BERT directory when on branch "ibert" rather than "ibert-base" as this needs access to the
# QUANTIZED models, rather than the normal bert model. To convert to bert model, see comments on the "from_pretrained" function.
# before running, change the paths to be compatible with your system, as they are not portable

# USAGE
# This script creates directories and fills them with two documents. The first is a text file that describes 
# the TSEL 0 (transformer sentence encoder layer 0) layer and submodules, followed by the names of all of the weights in the submodules.
# The names are used to access numpy arrays in the second file, the numpy zip. In this file, all internal weights and biases 
# are stored to be loaded by another python program, particularly when making an HLS model of the layers. Examples of loading
# and access of these numpy arrays are in the file "verify_model_weights.py". Use the names of the weights in the text file.


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
 

def print_module_info(module, f): #print the model heirarchy, then print the state dict keys
    string = "%s \n" % (module) #as a python noob, this is how I grab the string version of an object
    f.write(string)
    s = "%s \n" % module.state_dict().keys()
    s = s.replace(", ", "\n")
    f.write(s)

def get_TSEL_0(roberta): #gets Transformer Sentence Encoder Layer 0
    for name,module in roberta.named_modules():
        if(name == "model.encoder.sentence_encoder.layers.0"): #this is the first TransformerSentenceEncoderLayer
        #note, to add more, print name to a file and add the desired layer as criterion here
            return module

def save_model_weights(model, fname): #where model is a TransformerSentenceEncoderLayer
    dict_to_pack = model.state_dict()
    for key, value in dict_to_pack.items():
        if isinstance(value, torch.FloatTensor): #all parameters are stored as floatTensors
            array = value.numpy()
            dict_to_pack[key] = array #replacing the FloatTensor with a numpy array so I can zip it.
    np.savez(fname, **dict_to_pack)

def main():
    if os.path.isdir("export") == False:
        os.mkdir("export")
    for ckpt_path, model, testString in zip(CKPT_PATHS, MODELS, TESTNAMES):
        pathname = os.path.join("export", testString)
        if os.path.isdir(pathname) == False:
            os.mkdir(pathname)
            print("created " + pathname)
        print("working on " + testString)
        #Next we load a fine tuned and quantized roberta model. 
        roberta = RobertaModel.from_pretrained(ckpt_path, #RTEPATH when quantized, models/roberta.base/ when not
                                            checkpoint_file = "checkpoint_best.pt", #checkpoint_best.pt when quantized, model.pt when not
                                            data_name_or_path = model, 
                                            quant_mode = "symmetric", #symmetric when quantized, "none" when not
                                            force_dequant = "layernorm", #"layernorm" when quantized, "none" when not
                                            ) #data_name_or_path is need if using input data (inference)
        #I want to export just the first TSEL layer, because otherwise we are messing with GIGS of data
        transformerLayerModule = TransformerSentenceEncoderLayer #typing explicitly because I'm a C programmer
        transformerLayerModule = get_TSEL_0(roberta) #now this is the module we want information on
        textfile = "%s_modules.txt" % (testString)
        textfile = os.path.join("export", testString, textfile)
        textFD= open(textfile, "w") #OVERWRITE
        print_module_info(transformerLayerModule, textFD)
        #we printed keys to a text file, now print the numpy version of the weights,biases,etc tensors stored in the state dict
        numpyfile = "%s_state_dict.npz" % (testString)
        numpyfile = os.path.join("export", testString, numpyfile)
        save_model_weights(transformerLayerModule, numpyfile)
        
        textFD.close()
  
if __name__=="__main__":
    main()
