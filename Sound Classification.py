#!/usr/bin/env python
# coding: utf-8

# # Urban Sound Classification

# In[89]:





# In[95]:


import IPython.display as ipd
ipd.Audio('D:\\Urban Sound Classification\\train\\Train\\2022.wav')


# In[90]:


import numpy as np
import pandas as pd
import librosa 
import librosa.display
import glob 
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm import tqdm
import os


# In[2]:


test = pd.read_csv("D:\\Urban Sound Classification\\test\\test.csv", index_col = 0)
train = pd.read_csv("D:\\Urban Sound Classification\\train\\train.csv", index_col = 0)
exclude = pd.read_csv("C:\\Users\\Zara\\Desktop\\exclude.csv", index_col = 0)
exclude_test = pd.read_csv("C:\\Users\\Zara\\Desktop\\exclude_test.csv", index_col = 0)


# In[3]:


#masking mismatched or missing indexes from the train and test csv files

a_index = train.index
b_index = exclude.index
mask = ~a_index.isin(b_index)
train = train.loc[mask]
train = pd.DataFrame(train)
train = train.reset_index()
train = train.reset_index(drop = True)
train = pd.DataFrame(train)
train.head(7)


# In[4]:



a_index = test.index
b_index = exclude_test.index
mask = ~a_index.isin(b_index)
test = test.loc[mask]
test = pd.DataFrame(test)
test = test.reset_index()
test = test.reset_index(drop = True)
test = pd.DataFrame(test)
test.head(7)


# In[93]:


y, sr = librosa.load('D:\\Urban Sound Classification\\train\\Train\\2022.wav')


# In[94]:


plt.figure(figsize=(12, 4))
plt.title('Waveplot: Children Playing')
librosa.display.waveplot(y, sr=sr)


# In[75]:


#counts of items in each category

train['Class'].value_counts()
plt.bar(train['Class'].value_counts().index, train['Class'].value_counts(), align='center', data=train)
plt.title('Class Distribution')
plt.xticks(rotation='vertical')
plt.xlabel('Class')
plt.ylabel('Frequency')
print('Class Distribution:\n', train['Class'].value_counts())


# In[82]:


#MFCC of children playing
mfccs = librosa.feature.mfcc(y, sr, n_mfcc=40)
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()


# In[81]:


#melspectrogram of children playing
melspectrogram =librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000)
plt.figure(figsize=(10,4))
librosa.display.specshow(librosa.power_to_db(melspectrogram,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

#For explanation purpose.
#this function outputs the indexes of the files that cannot be read or are corrupt
#the output of the function was used to create the exclusion and exclusion_test files

def parser(row):
   # function to load files and extract features
    file_name = os.path.join(os.path.abspath(data_dir),str(row.ID)+'.wav')

   # handle exception to check if there isn't a file which is corrupted
    try:
      # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None
    

err = train.apply(parser, axis=1)
err.columns = ['feature', 'label']
# In[5]:


#extracting features from train set
#taking a mean of the mfccs

features=[]
labels=[]
for i in range(len(train)):
    data_dir = "D:\\Urban Sound Classification\\train\\Train\\"
    filename = os.path.join(os.path.abspath(data_dir)+'\\'+str(train.ID[i])+'.wav')
    x, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    features.append(np.mean(librosa.feature.mfcc(x, sr=sample_rate, n_mfcc=40).T,axis=0))
    labels.append(train.Class[i])


# In[6]:


#extracting features from test set
features_test=[]
for i in range(len(test)):
    test_dir = "D:\\Urban Sound Classification\\test\\Test\\"
    filename=os.path.join(os.path.abspath(test_dir)+'\\'+str(test.ID[i])+'.wav')
    x, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    features_test.append(np.mean(librosa.feature.mfcc(x, sr=sample_rate, n_mfcc=40).T,axis=0))


# In[7]:


#label encoding the categorical data (labels)
#making test and train

x=np.array(features)
x_test=np.array(features_test)

#encoding the labels
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))


# In[8]:


print('Shape of Features(Train): ',x.shape)
print('Shape of Features(Test): ',x_test.shape)
print('Shape of Labels(Train): ',y.shape)


# # Nueral Network

# In[9]:


import tensorflow as tf

model1=tf.keras.models.Sequential([
    
    tf.keras.layers.Dense(256,activation='relu',input_shape=(40,)),
    #using dropout regularization to randomly drop nodes , reduce overfitting, and improve generalization
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    #10 nodes since we have 10 labels
    tf.keras.layers.Dense(10,activation='softmax')
])


# In[10]:


#using tensorflow callbacks to reduce the learning rate if loss does not improve

reduce =tf. keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')

model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history1=model1.fit(x,y, batch_size=32, epochs=100, validation_split=0.1,callbacks=[reduce])


# In[11]:


#plot of training and validation accuracy/loss per epoch

acc=history1.history['accuracy']
val_acc=history1.history['val_accuracy']
loss=history1.history['loss']
val_loss=history1.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.legend()
plt.show()


# In[13]:


nn_pred=model1.predict(x_test)
nn_max=np.argmax(nn_pred,axis=1)
nn_pred_test=le.inverse_transform(nn_max)


# In[14]:


proba_nn = model1.predict_proba(x_test).ravel()


# In[22]:


nn_test_label = pd.DataFrame({'ID':test['ID'].values})
nn_test_label.insert(loc=0, column='Class', value=nn_pred_test)


# In[24]:


nn_test_label.head()


# # CNN

# In[25]:


max_pad_len=174 #defining a maximum length for each column 

file_name = "D:\\Urban Sound Classification\\train\\Train\\8279.wav"
audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
mfcc=librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
print('Shape of feature matrix: ',mfcc.shape)
pad_width = max_pad_len - mfcc.shape[1]
print('Pad width: ',pad_width)
mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
print('Shape of feature matrix after padding: ',mfcc.shape)


# In[26]:


#CNN does not support varying column length.fix by setting max length and padding
features_padded=[]
labels=[]
pad=174 # defining max column length as 174 for variation 

for i in range(len(train)):
    file_name = "D:\\Urban Sound Classification\\train\\Train\\"
    filename=os.path.join(os.path.abspath(file_name)+'\\'+str(train.ID[i]) +'.wav')
    x, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfcc=librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40)
    padding = pad - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, padding)), mode='constant')
    features_padded.append(mfcc)
    labels.append(train.Class[i])


# In[27]:


features_test_padded=[]

for i in range(len(test)):
    file_name = "D:\\Urban Sound Classification\\test\\Test\\"
    filename=os.path.join(os.path.abspath(file_name)+'\\'+str(test.ID[i])+'.wav')
    x, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfcc=librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40)
    padding = pad - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, padding)), mode='constant')
    features_test_padded.append(mfcc)


# In[28]:


x_new=np.array(features_padded)  
x_new=x_new.reshape(x_new.shape[0], 40, 174, 1) #reshaping the array for the convolution layer input

x_new_test=np.array(features_test_padded)  
x_new_test=x_new_test.reshape(x_new_test.shape[0], 40, 174, 1)

#encoding the labels
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))


# In[29]:


print('Shape of Features(Train): ',x_new.shape)
print('Shape of Features(Test): ',x_new_test.shape)
print('Shape of Labels(Train): ',y.shape)


# In[30]:


import tensorflow as tf


model2=tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(40,174,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(10,activation='softmax')
])


# In[31]:


reduce =tf. keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')

model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history2=model2.fit(x_new,y, batch_size=32, epochs=100, validation_split=0.1,callbacks=[reduce])


# In[32]:



acc=history2.history['accuracy']
val_acc=history2.history['val_accuracy']
loss=history2.history['loss']
val_loss=history2.history['val_loss']

epochs=range(len(acc)) #No. of epochs

#Plot training and validation accuracy per epoch
import matplotlib.pyplot as plt
plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.legend()
plt.figure()

#Plot training and validation loss per epoch
plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.legend()
plt.show()


# In[44]:


cnn_pred=model2.predict(x_new_test)
cnn_max=np.argmax(cnn_pred,axis=1)
cnn_pred_test=le.inverse_transform(cnn_max)


# In[45]:


proba_cnn = model2.predict_proba(x_new_test).ravel()


# In[47]:


cnn_test_label= pd.DataFrame({'ID':test['ID'].values})
cnn_test_label.insert(loc=0, column='Class', value=cnn_pred_test)


# In[48]:


cnn_test_label.head()

