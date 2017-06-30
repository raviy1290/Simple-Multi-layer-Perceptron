import mxnet as mx
import numpy as np
import logging
#data_iter = mx.io.CSVIter(data_csv='wine_data.csv', data_shape=(14,), batch_size=30)
#mx.nd.save('nddata')

logging.basicConfig(level=logging.INFO)
sample_count = 178
train_count = 140
valid_count = sample_count - train_count

feature_count = 13
category_count = 3
batch=1

load = np.loadtxt('wine_data.csv', delimiter=',', dtype=np.float)

X = load[:, 1:]
Y = load[:, 0]

X = mx.nd.array(X)
Y = mx.nd.array(Y)

X_train = X
Y_train = Y

X_valid = mx.nd.array([
[13.29,1.97,2.68,16.8,102,3,3.23,.31,1.66,6,1.07,2.84,1270],
[13.72,1.43,2.5,16.7,108,3.4,3.67,.19,2.04,6.8,.89,2.87,1285],
[12.37,1.63,2.3,24.5,88,2.22,2.45,.4,1.9,2.12,.89,2.78,342],
[12.04,4.3,2.38,22,80,2.1,1.75,.42,1.35,2.6,.79,2.57,580],
[12.88,2.99,2.4,20,104,1.3,1.22,.24,.83,5.4,.74,1.42,530],
[12.81,2.31,2.4,24,98,1.15,1.09,.27,.83,5.7,.66,1.36,560]
])

Y_valid = mx.nd.array([1,1,2,2,3,3])

print (X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)

# Build network
data = mx.sym.Variable('data')
fc1 = mx.sym.FullyConnected(data, name='fc1', num_hidden=39)
relu1 = mx.sym.Activation(fc1, name='relu1', act_type="relu")

fc2 = mx.sym.FullyConnected(relu1, name='fc2', num_hidden=3)
out = mx.sym.SoftmaxOutput(fc2, name='softmax')
#out = mx.sym.LogisticRegressionOutput(fc2)
mod = mx.mod.Module(out)

# Build iterator
train_iter = mx.io.NDArrayIter(data=X_train,label=Y_train,batch_size=batch)

# Train model
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
mod.fit(train_iter, num_epoch=100)


pred_iter = mx.io.NDArrayIter(data=X_valid,label=Y_valid, batch_size=batch)
pred_count = valid_count

correct_preds = total_correct_preds = 0
print('batch [labels] [predicted labels]  correct predictions')
for preds, i_batch, batch in mod.iter_predict(pred_iter):
    label = batch.label[0].asnumpy().astype(int)
    print i_batch, label, pred_label, correct_preds
    total_correct_preds = total_correct_preds + correct_preds

print('Validation accuracy: %2.2f' % (1.0*total_correct_preds/pred_count))

