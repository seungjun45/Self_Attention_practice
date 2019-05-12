import torch
import pdb

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

def save_model_CBAM(path, CBAM_Adapter, epoch, optimizer=None):
    CA=[]
    bn1=[]
    SA=[]
    bn2=[]

    for CA_param in CBAM_Adapter.C_list:
        for i in range(2):
            CA.append(CA_param[i].state_dict())

    for bn1_param in CBAM_Adapter.bn_channel:
        bn1.append(bn1_param.state_dict())

    for SA_param in CBAM_Adapter.S_list:
        SA.append(SA_param.state_dict())

    for bn2_param in CBAM_Adapter.bn_spatial:
        bn2.append(bn2_param.state_dict())


    model_dict = {
            'epoch': epoch,
            'ResNet_state': CBAM_Adapter.ResNet.state_dict(),
            'Channel_Attetion': CA,
            'BN_channel':bn1,
            'Spatial_Attention' : SA,
            'BN_spatial':bn2
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)

def save_model_BASELINE(path, ResNet, epoch, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'ResNet_state': ResNet.state_dict(),
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)


def load_model(Adapter_path, Adapter, optimizer=None, type='SE'):


    print('loading Adapter from {}'.format(Adapter_path))
    Adapter_model=torch.load(Adapter_path)
    Adapter.ResNet.load_state_dict(Adapter_model['ResNet_state'])

    training_epoch=Adapter_model['epoch']
    print('loaded epoch : {}'.format(training_epoch))

    if(optimizer):
        optimizer.load_state_dict(Adapter_model['optimizer_state'])

    if(type=='SE'): # Squeeze and Excitation
        C_list_pretrained=Adapter_model['SqueezeExcite_state']
        C_list_pretrained=[ [C_list_pretrained[2*i], C_list_pretrained[2*i+1]] for i in range(len(C_list_pretrained)//2)]
        for (SE_saved, SE_in_use) in zip(C_list_pretrained, Adapter.C_list):
            SE_in_use[0].load_state_dict(SE_saved[0])
            SE_in_use[1].load_state_dict(SE_saved[1])
    elif(type == 'BAM' or type=='CBAM'):
        C_list_pre=Adapter_model['Channel_Attetion']
        C_list_pre = [[C_list_pre[2 * i], C_list_pre[2 * i + 1]] for i in
                             range(len(C_list_pre) // 2)]
        S_list_pre=Adapter_model['Spatial_Attention']
        bn_channel_pre=Adapter_model['BN_channel']
        bn_spatial_pre=Adapter_model['BN_spatial']
        if (type == 'BAM'):
            S_list_pre = [[S_list_pre[4* i], S_list_pre[4 * i + 1], S_list_pre[4 * i + 2], S_list_pre[4 * i + 3]] for i in
                          range(len(S_list_pre) // 4)]
            bn_spatial_pre = [[bn_spatial_pre[3*i], bn_spatial_pre[3*i+1], bn_spatial_pre[3*i+2]] for i in
                              range(len(bn_spatial_pre) // 3) ]

        for (C_Att_pre, S_Att_pre, C_bn_pre, S_bn_pre, C_Att, S_Att, C_bn, S_bn) in \
                zip(C_list_pre, S_list_pre, bn_channel_pre, bn_spatial_pre, Adapter.C_list, Adapter.S_list, Adapter.bn_channel, Adapter.bn_spatial):

            if(type == 'BAM'):
                for i in range(2):
                    C_Att[i].load_state_dict(C_Att_pre[i])
                C_bn.load_state_dict(C_bn_pre)

                for i in range(4):
                    S_Att[i].load_state_dict(S_Att_pre[i])

                for i in range(3):
                    S_bn[i].load_state_dict(S_bn_pre[i])

            elif(type == 'CBAM'):
                for i in range(2):
                    C_Att[i].load_state_dict(C_Att_pre[i])

                C_bn.load_state_dict(C_bn_pre)

                S_Att.load_state_dict(S_Att_pre)

                S_bn.load_state_dict(S_bn_pre)


    return Adapter, training_epoch, optimizer


__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0], torch.sum(correct).item()