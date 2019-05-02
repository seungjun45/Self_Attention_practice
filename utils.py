import torch

def save_model(path, ResNet, SqueezeExcite, epoch, optimizer=None):
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