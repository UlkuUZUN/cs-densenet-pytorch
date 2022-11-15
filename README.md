# cs-densenet-pytorch
Cycle-spinning (CS) method employement into densenet-pytorch application

The pytorch densenet re-implement of the [DenseNet classification](https://pytorch.org/hub/pytorch_vision_densenet/) with Cycle-Spinning (CS) method employement performance in real time.

## Extra information for cycle-spinning please visit: https://github.com/UlkuUZUN/Cycle-Spinning-NN

In the original code, the Cycle-Spinning (CS) method has been applied to the model.py file to work with 1 and 2 shifts in the first convolution process. In our experiments it is seen that CS method employement gets good results if it works with first convolution and small shifts up to 3 shifts.

<div align="center">
  <p>
  <img width="850" src="https://github.com/UlkuUZUN/assets/blob/main/cs-densenet.png">
  </p>
</div>

