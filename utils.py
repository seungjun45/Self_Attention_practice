import torch

def save_model_SE(path, ResNet, SqueezeExcite, epoch, optimizer=None):
    SqueezeExcite_states=[]
    for SE_param in SqueezeExcite:
        SqueezeExcite_states.append(SE_param[0].state_dict())
        SqueezeExcite_states.append(SE_param[1].state_dict())
    model_dict = {
            'epoch': epoch,
            'ResNet_state': ResNet.state_dict(),
            'SqueezeExcite_state': SqueezeExcite_states
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)

def save_model_BAM(path, BAM_Adapter, epoch, optimizer=None):
    CA=[]
    bn1=[]
    SA=[]
    bn2=[]

    for CA_param in BAM_Adapter.C_list:
        for i in range(2):
            CA.append(CA_param[i].state_dict())

    for bn1_param in BAM_Adapter.bn_channel:
        bn1.append(bn1_param.state_dict())

    for SA_param in BAM_Adapter.S_list:
        for i in range(4):
            SA.append(SA_param[i].state_dict())

    for bn2_param in BAM_Adapter.bn_spatial:
        for i in range(3):
            bn2.append(bn2_param[i].state_dict())


    model_dict = {
            'epoch': epoch,
            'ResNet_state': BAM_Adapter.ResNet.state_dict(),
            'Channel_Attetion': CA,
            'BN_channel':bn1,
            'Spatial_Attention' : SA,
            'BN_spatial':bn2
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)



def load_model(Adapter_path, Adapter, epoch, optimizer=None):
    SqueezeExcite_states=[]
    for SE_param in SqueezeExcite:
        SqueezeExcite_states.append(SE_param[0].state_dict())
        SqueezeExcite_states.append(SE_param[1].state_dict())

    print('loading Adapter from {}'.format(Adapter_path))
    Adapter_model=torch.load(Adapter_path)
    Adapter.ResNet.load_state_dict(Adapter_model['ResNet_state'])


    model_dict = {
            'epoch': epoch,
            'ResNet_state': ResNet.state_dict(),
            'SqueezeExcite_state': SqueezeExcite_states
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)