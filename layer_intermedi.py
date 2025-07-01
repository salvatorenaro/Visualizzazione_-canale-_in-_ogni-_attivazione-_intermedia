#Se siete arrivati a questo punto avete una conoscenza avanzata sul machine-learning
#Quello che vedrete ora é : La visualizzazione dei layer intermedi serve a capire cosa sta “pensando” la rete neurale quando classifica un’immagine.
#Praticamente e come entrare all'interno della rete neurale

#Questa pratica lho usato per il dataset mnist ma potete impiegarlo anche per il modello di machine learning dogs_vs_cats

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#  SALVATORE NARO



# Caricamento del Dataset Mnist

#   Set di addestramento             Set di test  
#Featuers       Labels         featuers     Labels
   
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()#->Contiene imaggini di cifre scritte a mano dallo 0 al 9

# Preprocessing dei dati
#Set di addestramento:60000 immagini di 28x28 pixel a scala di grigi con un intensivita di pixel tra 0 e 255
train_images = train_images.reshape(60000,28,28,1).astype('float32')/255 #-> Disponiamo le imaggini in un array di 60000 righe e 784 colonne (28*28) e le normalizziamo tra 0 e 1
train_labels = to_categorical(train_labels,num_classes=10)#-> Ogni etichetta viene convertita in un vettore con 10 elementi(0-9)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Set di test:10000 immagini di 28x28 pixel a scala di grigi con un intensivita di pixel tra 0 e 255
test_images = test_images.reshape(10000,28,28,1).astype('float32')/255 #-> Disponiamo le imaggini in un array di 10000 righe e 784 colonne (28*28) e le normalizziamo tra 0 e 1
test_labels = to_categorical(test_labels,num_classes=10)#-> Ogni etichetta viene convertita in un vettore con 10 elementi(0-9)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creazione del Modello
model = models.Sequential()#>Creiamo un modello sequenziale
model.add(layers.Conv2D(32,(3,3),(1,1),activation='relu',padding='same',input_shape=(28,28,1)))#->Aggiungiamo un layer convuluzionale con 32 filtri di dimensione 3,3 (standard), stride di 1 (sposta di 1 pixel in modo da analizzare ogni pixel), attivazione ReLu, padding 'same' (standard), e input_shape di 28x28 pixel con 1 canale (scala di grigi)
model.add(layers.MaxPooling2D((2,2),padding='valid'))#->Aggiungiamo un layer di pooling che riduce le dimesioni dell'immagine di 2x2 (standard)
model.add(layers.Conv2D(64,(3,3),(1,1), activation='relu', padding='same'))#->Aggiungiamo un'altro layers convuluzionale con 64 filtri
model.add(layers.MaxPooling2D((2,2),padding='valid'))#->Aggiungiamo un'altro layer di pooling che riduce le dimesioni dell'immagine di 2x2 (standard)
model.add(layers.Conv2D(128,(3,3),(1,1), activation='relu', padding='same'))#->Aggiungiamo un'altro layers convuluzionale con 128 filtri praticamente stiamo aumentando il doppio ogni volta di filtri
model.add(layers.Flatten())#->Aggiungiamo un layer  flatten che appiattisce il vettore 2d in un vettore 1d cosi possiamo passare a layer densamente connessi tra loro
model.add(layers.Dense(128,activation='relu'))#->Aggiungiamo un layer densamente connesso con 128 neuroni(unita nascoste) e attivazione ReLu
model.add(layers.Dropout(0.5))#->Aggiungiamo un layer di dropout che elimina il 50% dei neuroni per evitare l'overfitting(Il fenome in cui il nostro modello apprende benissimo i dati di addestramento ma non impara i dati che non ha mai visto prima)
model.add(layers.Dense(10,activation='softmax'))#->Aggiungiamo un layer  con 10 neuroni (uno per ogni cifra) e attivazione softmax che restituisce una probabilita per ogni cifra
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Compilazione del modello
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])#->Utilizziamo come ottimizatore(Il meccanisco con la quale la rete si aggiornera su se stessa in base alla funzione obiettivo) rmsprop , funzione obiettivo (serve per valutare le proprie prestazioni sui dati di addestramento) categorical_crossentropy, metrica accuray 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Archittettura del modello
model.summary()#->Visualizziamo l'architettura del modello
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Addestramento del modello
history = model.fit(train_images,train_labels,epochs=10,batch_size=64,validation_split=0.2)#->Epochs(numero di iterazioni complete nella quale aggiorna i propri pesi) 10, batch_size (i dati che la rete utilizza per apprendere) 64, verbose(per visualizzare l'andamento) 1 
result = model.evaluate(test_images,test_labels)#->Valutiamo il modello sui dati di test , loss e la vicinanza della rete sui dati di addestramento mentre acc Ã¨ l'accuratezza del modello sui dati di addestramento
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(f"Test Loss: {result[0]}, Test Accuracy: {result[1]}")#->Visualizziamo 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Visualizzazione di ogni canale in ogni attivazione intermedia
layers_output = [layer.output for layer in model.layers[:8]] #Prendiamo i primi 8
print(layers_output)

activation_model = models.Model(inputs=model.inputs[0], outputs=layers_output) #Creiamo un modello che invece di darci il risultato finale ci fornisce l'output dei layer intermedi

sample_image = test_images[0] #Prendo la prima immagine
sample_image = np.expand_dims(sample_image, axis=0) #la rete di aspetta un batch e anche 1 sola immagine deve avere il batch quindi usiamo questo comando per passare da -> 28,28,1  a 1,28,28,1 

activations = activation_model.predict(sample_image) #ottiamo l'output di ogni layer

first_layer_attivato = activations[0] #Visualizziamo il primo layer
print(first_layer_attivato)

layer_name = []#conservermo i nomi
for layer in model.layers[:8]:#prendiamo 8 layer
    layer_name.append(layer.name) #li conserviamo nella lista
print(layer_name)

images_per_row = 8 #filtri per riga
for layer_name,layer_activation in zip(layer_name,activations):
    if len(layer_activation.shape) != 4: #controllo per saltare i layer densi flatten e dropout
        continue
    n_features = layer_activation.shape[-1] #filtro 
    print(n_features)
    size = layer_activation.shape[1] #dimensione
    print(size)
    n_cols = n_features // images_per_row #divisione fra i filtri e i filtri per riga quidi fa 32/8 64/8 128/8
    print(n_cols)
    display_grid = np.zeros((size*n_cols,images_per_row*size)) #matrice di 0 con le dimensioni 28*4, 8*28
    print(display_grid)
    for col in range(n_cols):#per ogni riga
        for row in range(images_per_row):#per ogni colonna
            channel_index = col * images_per_row + row #calcolo quale filtro
            if channel_index >= n_features:#controllo dei filtri 
                continue
            channel_image = layer_activation[0, :, :, channel_index]
            channel_image -= channel_image.mean() #centra i valori attorno lo 0
            channel_image /= (channel_image.std() + 1e-5) #normalizza
            channel_image *= 64 #il contrasto
            channel_image += 128 #sposta nel range positivo
            channel_image = np.clip(channel_image, 0, 255).astype('uint8') #porto i valori tra lo 0,255
            display_grid[col * size : (col + 1) * size,
                        row * size : (row + 1) * size] = channel_image #posizione corretta del filtro dell'immagine
    #disegnamo il grafico
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

