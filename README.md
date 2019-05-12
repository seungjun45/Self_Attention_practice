# Self_Attention_practice
Pytorch implementation of Squeeze_and_Excitation, Bottleneck_Attention_Module, and Convolutional_Block_Attention_Module : for EE898 class, KAIST
</br></br>
# Experiment environment ==> pytorch 0.4 (pytorch 1.0 not supported)
</br>
python eval.py --model_path model-299.pth --type SE --model res50</br>
top1: 38.87</br>
top5: 15.52</br>
</br>
python eval.py --model_path model-299.pth --type SE --model res34</br>
top1: 42.49</br>
top5: 18.84</br>
</br>
python eval.py --model_path model-299.pth --type BAM --model res50</br>
top1:  41.95</br>
top5: 17.86</br>
</br>
python eval.py --model_path model-299.pth --type BAM --model res34</br>
top1: 42.36</br>
top5: 18.27</br>
</br>
python eval.py --model_path model-299.pth --type CBAM --model res50</br>
top1: 41.38</br>
tpo5: 17.71</br>
</br>
python eval.py --model_path model-299.pth --type CBAM --model res34</br>
top1: 43.03</br>
top5: 19.65</br>
</br>
python eval.py --model_path model-299.pth --type Baseline --model res50</br>
top1: 39.57</br>
top5: 15.77</br>
</br>
python eval.py --model_path model-299.pth --type Baseline --model res34</br>
top1: 41.57</br>
top5: 18.12</br>
</br>
# Conclusion..
Am I doing something wrong??