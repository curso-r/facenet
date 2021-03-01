import torch

torch.save(dict(torch.load("inst/onet.pt")), 'inst/onet_reload.pt', _use_new_zipfile_serialization = True)
torch.save(dict(torch.load("inst/pnet.pt")), 'inst/pnet_reload.pt', _use_new_zipfile_serialization = True)
torch.save(dict(torch.load("inst/rnet.pt")), 'inst/rnet_reload.pt', _use_new_zipfile_serialization = True)

