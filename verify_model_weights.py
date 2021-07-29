import numpy as np
print("*** RTE ***")
npzfile = np.load("export/RTE/RTE_state_dict.npz")
print(npzfile["self_attn.k_proj.weight"])
print("***")


print("*** SST-2 ***")
npzfile = np.load("export/SST-2/SST-2_state_dict.npz")
print(npzfile["self_attn.k_proj.weight"])
print("***")


print("*** MNLI ***")
npzfile = np.load("export/MNLI/MNLI_state_dict.npz")
print(npzfile["self_attn.k_proj.weight"])
print("***")


print("*** QNLI ***")
npzfile = np.load("export/QNLI/QNLI_state_dict.npz")
print(npzfile["self_attn.k_proj.weight"])
print("***")


print("*** CoLA ***")
npzfile = np.load("export/CoLA/CoLA_state_dict.npz")
print(npzfile["self_attn.k_proj.weight"])
print("***")


print("*** QQP ***")
npzfile = np.load("export/QQP/QQP_state_dict.npz")
print(npzfile["self_attn.k_proj.weight"])
print("***")


print("*** MRPC ***")
npzfile = np.load("export/MRPC/MRPC_state_dict.npz")
print(npzfile["self_attn.k_proj.weight"])
print("***")


print("*** STS-B ***")
npzfile = np.load("export/STS-B/STS-B_state_dict.npz")
print(npzfile["self_attn.k_proj.weight"])
npzfile = np.load("export/MRPC/MRPC_state_dict.npz")
print("***")

print(type(npzfile))
for name in npzfile:
    print(npzfile[name].shape)