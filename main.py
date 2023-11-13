from src.model import NNmodel
from src.dense_layer import Layer
from src.utils import *
from src.dataset import load_dataset

if __name__=='__main__':
    x_train, y_train, x_test, y_test, classes = load_dataset()

    epochs = 1501
    inputs = x_train.shape[0]

    model = NNmodel(epochs=epochs)
    model.add(Layer(inputs=inputs, neurons=5, activation='relu'))
    model.add(Layer(inputs=5, neurons=1, activation='sigmoid'))
    model.fit(x_train=x_train, y_train=y_train)

    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    train_acc = getAccuracy(Y=y_train, predictions=train_preds)
    test_acc = getAccuracy(Y=y_test, predictions=test_preds)

    print(f'Train Accuracy - {train_acc}\nTest Accuracy - {test_acc}')