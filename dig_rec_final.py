import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random as rd
##tf.__version__
data = tf.keras.datasets.mnist #28x28 el yazısı rakamlar 0-9
(x_train, y_train), (x_test, y_test) = data.load_data()
print(x_train.shape) #eğitim verilerinin formatı

# ilk rakam
plt.imshow(x_train[0], cmap = "gray", vmin = 0, vmax = 255)
plt.show()
print(x_train[0][:10,:10]) #normalizasyondan önce

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(x_train[i], cmap='gray', interpolation='none')
  plt.title("Rakam: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])

# 784 pikselin 0-256 aralığıdaki dağılımı
digit = 0
fig = plt.figure()
plt.subplot(2,1,1)
plt.imshow(x_train[digit], cmap='gray', interpolation='none')
plt.title("Digit: {}".format(y_train[digit]))
plt.xticks([])
plt.yticks([])
plt.subplot(2,1,2) 
plt.hist(x_train[digit].reshape(784))
plt.title("Piksel Değerlerinin Dağılımı")

x_train = tf.keras.utils.normalize(x_train, axis = 1) #satırların bütün sütunlarını 0-1 aralığında normalize eder
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential() #sequential modeli
 
                                    
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #girdi katmanı rakamın bütün pikselleri
model.add(tf.keras.layers.Dense(512, activation =  tf.nn.relu)) # Gizli katman
model.add(tf.keras.layers.Dropout(0.2)) # overfitting engellemek için %2 sini siliyoruz
model.add(tf.keras.layers.Dense(512, activation =  tf.nn.relu)) # Gizli katman 2
model.add(tf.keras.layers.Dropout(0.2)) # overfitting engellemek için %2 sini siliyoruz
model.add(tf.keras.layers.Dense(10, activation =  tf.nn.softmax)) #çıktı katmanı 10 rakam olduğu için 10 nöron var

model.compile(loss= 'sparse_categorical_crossentropy',
             metrics= ['accuracy'],optimizer= 'adam',)

plt.imshow(x_train[0], cmap = "gray", vmin = 0, vmax = 1)
plt.show()

print(x_train[0][:10,:10]) #verinin normalizasyondan sonraki hali

# modelin eğitimi ve sürecin kaydı
history = model.fit(x_train, y_train,
          batch_size=128, epochs=20, verbose = 2,
          validation_data=(x_test, y_test))

# eğitim sürecinin grafiksel yorumu
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Modelin Başarısı')
plt.ylabel('Doğruluk')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Modelin Kayıpları')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Test'], loc='upper right')

plt.tight_layout()

model.save("dig_rec.model")

new_model = tf.keras.models.load_model('dig_rec.model')

predictions = new_model.predict([x_test])

print("Tahmin Sonucu (Output Layer)")
print(predictions[0]) #düzenlenmediği için tahmin değeri dizi formatında, one hot encoding
print()
print("Tahmin Sonucu (Düzenlenmiş): ", np.argmax(predictions[0]))
print("Gerçek değer: ", y_test[0])

testSize = 6
random_digits = np.arange((testSize))
print("Modelin Tahminleri")
for i in range(testSize):
    random_digit = rd.randrange(0 , 10000)
    random_digits[i] = random_digit
    print(random_digit,"satırındaki rakam:",np.argmax(predictions[random_digit]))

fig = plt.figure()
for j in range(testSize):
  plt.subplot(3,3,j+1)
  plt.tight_layout()
  plt.imshow(x_test[random_digits[j]], cmap = "gray", interpolation='none', vmin= 0, vmax = 0.25)
  plt.title("Rakam: {}".format(y_test[random_digits[j]]))
  plt.xticks([])
  plt.yticks([])

pred_array = np.arange(10000)
cnt = 0
for k in predictions:
    pred_array[cnt] = np.argmax(k)
    cnt+= 1
pred_array


from sklearn.metrics import accuracy_score
acc_Score = accuracy_score(y_test, pred_array)
acc_Score


k = 0
cnt = 0
total_lost = int((100*(100 - acc_Score*100))) + 1
lost_digits = np.arange((total_lost))
for i in pred_array:
    if y_test[cnt] != i:
        lost_digits[k] = cnt
        k+= 1
    cnt+=1

for i in range(testSize):
    random_digit = rd.randrange(0 , lost_digits.size)
    index = lost_digits[random_digit]
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(x_test[index], cmap = "gray", interpolation='none', vmin= 0, vmax = 0.25)
    plt.title("Tahmin Edilen: {}\n Gerçek değer: {}".format(pred_array[index],y_test[index]))
    plt.xticks([])
    plt.yticks([])

from tensorflow.keras.utils import plot_model
plot_model(model, to_file="model_plot.png", show_shapes = True, show_layer_names = True, rankdir = "LR")
