from keras.layers import Conv3D,MaxPool3D,Conv3DTranspose,Input,Concatenate
from keras.models import Model

def model():
    IMAGE_ORDERING =  "channels_first"
    ## ENCODER T1
    img_input_t1 = Input(shape=(1,240,240,48), name = "T1")
    conv_1_t1 = Conv3D(filters=8,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_1_T1",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(img_input_t1)
    conv_11_t1 = Conv3D(filters=8,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_11_T1",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_1_t1)
    maxpool_1_t1 = MaxPool3D(name = "MAXPOOL3D_1_T1",data_format=IMAGE_ORDERING)(conv_11_t1)
    conv_2_t1 = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_2_T1",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(maxpool_1_t1)
    conv_21_t1 = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_21_T1",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_2_t1)
    maxpool_2_t1 = MaxPool3D(name = "MAXPOOL3D_2_T1",data_format=IMAGE_ORDERING)(conv_21_t1)
    conv_3_t1 = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_3_T1",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(maxpool_2_t1)
    conv_31_t1 = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_31_T1",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_3_t1)
    conv_32_t1 = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_32_T1",dilation_rate=(3, 3, 3),data_format=IMAGE_ORDERING)(conv_31_t1)
    maxpool_3_t1 = MaxPool3D(name = "MAXPOOL3D_3_T1",data_format=IMAGE_ORDERING)(conv_32_t1)

    ## ENCODER FLAIR
    img_input_FLAIR = Input(shape=(1,240,240,48), name = "FLAIR")
    conv_1_FLAIR = Conv3D(filters=8,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_1_FLAIR",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(img_input_FLAIR)
    conv_11_FLAIR = Conv3D(filters=8,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_11_FLAIR",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_1_FLAIR)
    maxpool_1_FLAIR = MaxPool3D(name = "MAXPOOL3D_1_FLAIR",data_format=IMAGE_ORDERING)(conv_11_FLAIR)
    conv_2_FLAIR = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_2_FLAIR",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(maxpool_1_FLAIR)
    conv_21_FLAIR = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_21_FLAIR",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_2_FLAIR)
    maxpool_2_FLAIR = MaxPool3D(name = "MAXPOOL3D_2_FLAIR",data_format=IMAGE_ORDERING)(conv_21_FLAIR)
    conv_3_FLAIR = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_3_FLAIR",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(maxpool_2_FLAIR)
    conv_31_FLAIR = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_31_FLAIR",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_3_FLAIR)
    conv_32_FLAIR = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_32_FLAIR",dilation_rate=(3, 3, 3),data_format=IMAGE_ORDERING)(conv_31_FLAIR)
    maxpool_3_FLAIR = MaxPool3D(name = "MAXPOOL3D_3_FLAIR",data_format=IMAGE_ORDERING)(conv_32_FLAIR)

    ## ENCODER IR
    img_input_IR = Input(shape=(1,240,240,48), name = "IR")
    conv_1_IR = Conv3D(filters=8,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_1_IR",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(img_input_IR)
    conv_11_IR = Conv3D(filters=8,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_11_IR",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_1_IR)
    maxpool_1_IR = MaxPool3D(name = "MAXPOOL3D_1_IR",data_format=IMAGE_ORDERING)(conv_11_IR)
    conv_2_IR = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_2_IR",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(maxpool_1_IR)
    conv_21_IR = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_21_IR",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_2_IR)
    maxpool_2_IR = MaxPool3D(name = "MAXPOOL3D_2_IR",data_format=IMAGE_ORDERING)(conv_21_IR)
    conv_3_IR = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_3_IR",dilation_rate=(1, 1, 1),data_format=IMAGE_ORDERING)(maxpool_2_IR)
    conv_31_IR = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_31_IR",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(conv_3_IR)
    conv_32_IR = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_32_IR",dilation_rate=(3, 3, 3),data_format=IMAGE_ORDERING)(conv_31_IR)
    maxpool_3_IR = MaxPool3D(name = "MAXPOOL3D_3_IR",data_format=IMAGE_ORDERING)(conv_32_IR)

    ## Concatenate ALL
    concat_all = Concatenate(axis = 1)([maxpool_3_t1,maxpool_3_FLAIR,maxpool_3_IR])
    conv_all = Conv3D(filters=64,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_all",data_format=IMAGE_ORDERING)(concat_all)

    ## DECODER
    convt_1 = Conv3DTranspose(32,kernel_size=(2,2,2),strides=(2,2,2),name = "CONV3DT_1",activation='relu',data_format=IMAGE_ORDERING)(conv_all)
    concat_1 = Concatenate(axis=1)([convt_1,conv_32_t1,conv_32_FLAIR,conv_32_IR])
    conv_4 = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_4",data_format=IMAGE_ORDERING)(concat_1)
    conv_41 = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_41",data_format=IMAGE_ORDERING)(conv_4)
    convt_2 = Conv3DTranspose(16,kernel_size=(2,2,2),strides=(2,2,2),name = "CONV3DT_2",activation='relu',data_format=IMAGE_ORDERING)(conv_41)
    concat_2 = Concatenate(axis=1)([convt_2,conv_21_t1,conv_21_FLAIR,conv_21_IR])
    conv_5 = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_5",data_format=IMAGE_ORDERING)(concat_2)
    conv_51 = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_51",data_format=IMAGE_ORDERING)(conv_5)
    convt_3 = Conv3DTranspose(4,kernel_size=(2,2,2),strides=(2,2,2),name = "CONV3DT_3",activation='relu',data_format=IMAGE_ORDERING)(conv_51)
    conv_6 = Conv3D(filters=1,kernel_size=(3, 3, 3),padding='same',activation='sigmoid',name = "CONV3D_6",data_format=IMAGE_ORDERING)(convt_3)
    return Model([img_input_t1,img_input_FLAIR,img_input_IR], conv_6)

def model_thresholding():
    IMAGE_ORDERING =  "channels_first"
    img_input = Input(shape=(1,240,240,48))
    conv_1 = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_1",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(img_input)
    maxpool_1 = MaxPool3D(name = "MAXPOOL3D_1",data_format=IMAGE_ORDERING)(conv_1)
    conv_2 = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_2",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(maxpool_1)
    maxpool_2 = MaxPool3D(name = "MAXPOOL3D_2",data_format=IMAGE_ORDERING)(conv_2)
    conv_3 = Conv3D(filters=32,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_3",dilation_rate=(2, 2, 2),data_format=IMAGE_ORDERING)(maxpool_2)

    convt_1 = Conv3DTranspose(16,kernel_size=(2,2,2),strides=(2,2,2),name = "CONV3DT_1",activation='relu',data_format=IMAGE_ORDERING)(conv_3)
    concat_1 = Concatenate(axis=1)([convt_1,conv_2])
    conv_4 = Conv3D(filters=16,kernel_size=(3, 3, 3),padding='same',activation='relu',name = "CONV3D_4",data_format=IMAGE_ORDERING)(concat_1)
    convt_2 = Conv3DTranspose(4,kernel_size=(2,2,2),strides=(2,2,2),name = "CONV3DT_2",activation='relu',data_format=IMAGE_ORDERING)(conv_4)
    concat_2 = Concatenate(axis=1)([convt_2,conv_1])
    conv_5 = Conv3D(filters=1,kernel_size=(3, 3, 3),padding='same',activation='sigmoid',name = "CONV3D_5",data_format=IMAGE_ORDERING)(concat_2)
    return Model(img_input, conv_5)
    concat_2 = Concatenate(axis=1)([convt_2,conv_1])
    conv_5 = Conv3D(filters=1,kernel_size=(3, 3, 3),padding='same',activation='sigmoid',name = "CONV3D_5",data_format=IMAGE_ORDERING)(concat_2)
    return Model(img_input, conv_5)
