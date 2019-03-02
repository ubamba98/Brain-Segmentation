from utils.utils import data_train, data_val
from utils.model import model, model_thresholding
from utils.losses import dice_coefficient, dice_loss
from utils.preprocess import *
import pickle
import keras

## VARIABLES
ROOT = "./" ##BASE PATH TO MRBrainS18
LABEL = "Basal ganglia" ##LABEL TO TRAIN FOR
EPOCHS = 400 ##NUMBER OF EPOCHS


T1path, FLAIRpath, IRpath, segpath = data_train(root = ROOT)
T1_val, FLAIR_val, IR_val, segm_val = data_val(root = ROOT)

labels = {
    "Cortical gray matter" : 1,
    "Basal ganglia" : 2,
    "White matter" : 3,
    "White matter lesions" : 4,
    "Cerebrospinal fluid in the extracerebral space" : 5,
    "Ventricles" : 6,
    "Cerebellum" : 7,
    "Brain stem" : 8
}

label = labels[LABEL]

if label in [1,3,5]:
    print("TRAINING ON THRESHOLDING MODEL...")
    print("LOADING DATA...")
    X = []
    y = []
    if label == 5:
        for T1_,seg_ in zip(T1path, segpath):
            T1 = get_data_with_skull_scraping(T1_)
            y.append(np.array(get_data(seg_)==5).astype(np.uint8)[None,...])
            X.append(np.array((T1>=10) & (T1<110)).astype(np.uint8)[None,...])#<-Works better
        X = np.array(X)
        y = np.array(y)
        T1 = get_data_with_skull_scraping(T1_val)
        X_val = np.array((T1>=10) & (T1<110)).astype(np.uint8)[None,None,...]
        y_val = np.array(get_data(segm_val)==5).astype(np.uint8)[None,...]
    elif label == 3:
        for T1_,seg_ in zip(T1path, segpath):
            T1 = get_data_with_skull_scraping(T1_)
            y.append(np.array(get_data(seg_)==3).astype(np.uint8)[None,...])
            X.append(np.array(T1>=150).astype(np.uint8)[None,...])
        X = np.array(X)
        y = np.array(y)
        T1 = get_data_with_skull_scraping(T1_val)
        X_val = np.array(T1>=150).astype(np.uint8)[None,None,...]
        y_val = np.array(get_data(segm_val)==3).astype(np.uint8)[None,...]
    else:
        for T1_,seg_ in zip(T1path, segpath):
            T1 = get_data_with_skull_scraping(T1_)
            y.append(np.array(get_data(seg_)==1).astype(np.uint8)[None,...])
            X.append(np.array((T1>=80) & (T1<160)).astype(np.uint8)[None,...])
        X = np.array(X)
        y = np.array(y)
        T1 = get_data_with_skull_scraping(T1_val)
        X_val = np.array((T1>=80) & (T1<160)).astype(np.uint8)[None,None,...]
        y_val = np.array(get_data(segm_val)==1).astype(np.uint8)[None,...]

    print("STARTING TRAINING...")
    model_ = model_thresholding()
    model_.compile('adam',dice_loss,[dice_coefficient])
    model_.summary()
    history = model_.fit(X,y,validation_data=(X_val,y_val),epochs=EPOCHS,callbacks=[keras.callbacks.ModelCheckpoint(
                  'weights/label'+str(label)+'/Model.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
                  monitor='val_dice_coefficient',
                  verbose=0,
                  save_best_only=True,
                  save_weights_only=False,
                  mode='max',
                  period=1)])

else:
    print("LOADING DATA...")
    X_T1 = []
    X_FLAIR = []
    X_IR = []
    y = []
    for T1_,FLAIR_,IR_,seg_ in zip(T1path, FLAIRpath, IRpath, segpath):
        T1 = histeq(to_uint8(get_data_with_skull_scraping(T1_)))
        IR = IR_to_uint8(get_data(IR_))
        FLAIR = to_uint8(get_data(FLAIR_))
        y.append(np.array(get_data(seg_)==label).astype(np.uint8)[None,...])
        X_T1.append(T1[None,...])
        X_IR.append(IR[None,...])
        X_FLAIR.append(FLAIR[None,...])
    X_T1 = np.array(X_T1)
    X_FLAIR = np.array(X_FLAIR)
    X_IR = np.array(X_IR)
    y = np.array(y)
    X_T1_val = histeq(to_uint8(get_data_with_skull_scraping(T1_val)))[None,None,...]
    X_FLAIR_val = to_uint8(get_data(FLAIR_val))[None,None,...]
    X_IR_val = IR_to_uint8(get_data(IR_val))[None,None,...]
    y_val = np.array(get_data(segm_val)==CLASS).astype(np.uint8)[None,...]
    print("STARTING TRAINING...")
    model_ = model()
    model_.compile('adam',dice_loss,[dice_coefficient])
    model_.summary()
    history = model_.fit([X_T1,X_FLAIR,X_IR],y=y,validation_data=([X_T1_val,X_FLAIR_val,X_IR_val],y_val),epochs=EPOCHS,callbacks=[keras.callbacks.ModelCheckpoint(
                      'weights/label'+str(label)+'/Model.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
                      monitor='val_dice_coefficient',
                      verbose=0,
                      save_best_only=True,
                      save_weights_only=False,
                      mode='max',
                      period=1
          )])
print("FINISHED TRAINING...")
print("Saving training history")
with open('history/trainHistoryDict'+str(label)+'.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
print("DONE")
