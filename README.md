# はじめての機械学習(自分でデータセットを作る編)のサンプルコード

## 概要

Zennの[はじめての機械学習(自分でデータセットを作る編)](https://zenn.dev/kaityo256/articles/my_first_machine_learning)のサンプルコードです。

## 実行方法

```sh
git clone https://github.com/kaityo256/my_first_ml.git
cd myfirst_ml
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow
```

Macの場合は仮想環境作成時に`Python3.11`を使う必要があることに注意。

```sh
python3.11 -m venv .venv 
source .venv/bin/activate
```

### MNIST

MNISTの訓練。

```sh
$ python3 mnist_train.py
(snip)
Epoch 1/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2640 - accuracy: 0.9252
Epoch 2/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1173 - accuracy: 0.9649
Epoch 3/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0807 - accuracy: 0.9762
Epoch 4/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0597 - accuracy: 0.9816
Epoch 5/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.0467 - accuracy: 0.9852
313/313 [==============================] - 1s 2ms/step - loss: 0.0815 - accuracy: 0.9746
Test Loss = 0.08145684003829956
Test Accuracy = 0.9746000170707703
```

学習済みモデルの読み込みと確認。

```sh
$ python3 mnist_load.py
prediction= 7 answer = 7
prediction= 2 answer = 2
prediction= 1 answer = 1
prediction= 0 answer = 0
prediction= 4 answer = 4
prediction= 1 answer = 1
prediction= 4 answer = 4
prediction= 9 answer = 9
prediction= 6 answer = 5
prediction= 9 answer = 9
prediction= 0 answer = 0
prediction= 6 answer = 6
prediction= 9 answer = 9
prediction= 0 answer = 0
prediction= 1 answer = 1
prediction= 5 answer = 5
prediction= 9 answer = 9
prediction= 7 answer = 7
prediction= 3 answer = 3
prediction= 4 answer = 4
```

### パイこね変換

乱数とパイこね変換の分類器を訓練。

```sh
$ python3 baker_train.py
Epoch 1/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.6435 - accuracy: 0.6104
Epoch 2/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.4758 - accuracy: 0.7699
Epoch 3/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.4047 - accuracy: 0.8117
Epoch 4/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3718 - accuracy: 0.8306
Epoch 5/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3527 - accuracy: 0.8410
313/313 [==============================] - 1s 1ms/step - loss: 0.3988 - accuracy: 0.8169
Test Loss = 0.3987952172756195
Test Accuracy = 0.8169000148773193
```

学習済みモデルの読み込みと確認。

```sh
$ python3 baker_load.py
4/4 [==============================] - 0s 2ms/step - loss: 0.3496 - accuracy: 0.8300
When everything is random
Test Loss = 0.34963053464889526
Test Accuracy = 0.8299999833106995
4/4 [==============================] - 0s 1ms/step - loss: 0.4128 - accuracy: 0.8400
When everything is baker map
Test Loss = 0.4127606153488159
Test Accuracy = 0.8399999737739563
```

## LICENSE

MIT
