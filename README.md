# MQ-ABR
Recommend Pytorch 1.5.0 (we don't test in other versions) and tenroboardX

All training data and Testing data have been prepared, and you can train the model just running


```
python multi_pytorch.py
```

The training process can be visualized by running

```
tensorboard --logdir=./results/
```

where the plot can be viewed at `localhost:6006` from a browser. 

Put up the issue if you have any questions.

If you want to test DNN using the tuning algorithm by adjusting the minimum frame rate, please by run `Minimum_frame_rate.py` under folder Frame_rate_test 
and then adjust the list of minimum frame rates in `test.py` to determine the minimum frame rate in the table.

