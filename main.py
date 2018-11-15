
import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans 
import random
from sklearn.preprocessing import MinMaxScaler

trainingpercent=80
ValidationPercent = 10
TestPercent = 10
M=4
C_Lambda = 0.03
IsSynthetic = False

"""Human observed dataset"""
samepairs=pd.read_csv('same_pairs.csv',header=0)
diffpairs=pd.read_csv('diffn_pairs.csv',header=0)
imgfeatures=pd.read_csv('HumanObserved-Features-Data.csv',index_col=0)

"""Gsc dataset"""
samegsc=pd.read_csv('same_pairs_gsc.csv',header=0)
samegsc = samegsc.loc[random.sample(range(samegsc.shape[0]),7900),:]
diffgsc=pd.read_csv('diffn_pairs_gsc.csv',header=0)
diffgsc = diffgsc.loc[random.sample(range(diffgsc.shape[0]),7900),:]
imggsc=pd.read_csv('GSC-Features.csv',header=0)
imggsc = imggsc.loc[random.sample(range(imggsc.shape[0]),7900),:]

"""Function for preprocessing data by concatenation"""
def concat(samepairs,diffpairs,imagefeatures,length,case):
    #calculating same pair features by inner join 
    halfmrg = pd.merge(samepairs, imagefeatures, how='inner', left_on=['img_id_A'], right_on=['img_id'])
    final_merge=pd.merge(halfmrg, imagefeatures, how='inner', left_on=['img_id_B'], right_on=['img_id'])
    final_merge=final_merge.drop(columns=['img_id_x', 'img_id_y'])
  
    #calculating different pair features by inner join 
    result_diff = pd.merge(diffpairs, imagefeatures, how='inner', left_on=['img_id_A'], right_on=['img_id'])
    final_merge_diff=pd.merge(result_diff, imagefeatures, how='inner', left_on=['img_id_B'], right_on=['img_id'])
    final_merge_diff=final_merge_diff.drop(columns=['img_id_x', 'img_id_y'])
    
    if(case==1):
        #condition for concatenation in case of human observed data
        concat_human = pd.concat([final_merge, final_merge_diff.take(np.random.permutation(len(final_merge_diff))[:length])])
        #shuffling data for increasing randomness
        concat_human=concat_human.iloc[np.random.permutation(len(concat_human))]
        concat_human = concat_human.reset_index(drop=True)
    else:
        #condition for concatenation in case of gsc data
        concat_human = pd.concat([final_merge,final_merge_diff])
        concat_human = concat_human.iloc[np.random.permutation(len(concat_human))]
        concat_human = concat_human.reset_index(drop=True)

    targetconcat=concat_human['target']
    concat_human=concat_human.drop(columns=['img_id_A','img_id_B','target'])
    #deleting feature columns whose values are 0
    concat_human.loc[(concat_human!= 0).any(axis=1)]
    return concat_human,targetconcat


def subtract(samepairs,diffpairs,imagefeatures,length,case):
    
    #calculating same pair features by inner join 
    resultofa = pd.merge(samepairs, imagefeatures, how='inner', left_on=['img_id_A'], right_on=['img_id'])
    resultofa=resultofa.drop(['img_id_A','img_id_B','target','img_id'],axis=1)
    resultofb = pd.merge(samepairs, imagefeatures, how='inner', left_on=['img_id_B'], right_on=['img_id'])
    resultofb=resultofb.drop(['img_id_A','img_id_B','target','img_id'],axis=1)
    sub_human=resultofa.sub(resultofb,fill_value=0)
    sub_human=np.abs(sub_human)
    sub_human['target']=1

    #calculating different pairs features by inner join 
    resultofdiffa = pd.merge(diffpairs, imagefeatures, how='inner', left_on=['img_id_A'], right_on=['img_id'])
    resultofdiffa=resultofdiffa.drop(['img_id_A','img_id_B','target','img_id'],axis=1)
    resultofdiffb = pd.merge(diffpairs, imagefeatures, how='inner', left_on=['img_id_B'], right_on=['img_id'])
    resultofdiffb=resultofdiffb.drop(['img_id_A','img_id_B','target','img_id'],axis=1)
    sub_humandiff=resultofdiffa.sub((resultofdiffb),fill_value=0)
    sub_humandiff=np.abs(sub_humandiff)
    sub_humandiff['target']=0

    if(case==1):
        #condition for subtraction in case of human observed data
        concat_humandiff = pd.concat([sub_human, sub_humandiff.take(np.random.permutation(len(sub_humandiff))[:length])])
        concat_humandiff=concat_humandiff.iloc[np.random.permutation(len(concat_humandiff))]
        concat_humandiff = concat_humandiff.reset_index(drop=True)
    else:
        #condition for subtraction in case of gsc data
        frames=[sub_human[:7900], sub_humandiff[:7900]]
        concat_humandiff = pd.concat(frames)
        concat_humandiff = concat_humandiff.iloc[np.random.permutation(len(concat_humandiff))]
        concat_humandiff = concat_humandiff.reset_index(drop=True)
    targetsub=concat_humandiff['target']
    concat_humandiff=concat_humandiff.drop(['target'],axis=1)
    #deleting feature columns whose values are 0
    concat_humandiff=concat_humandiff[(concat_humandiff!= 0).any(1)]
    return concat_humandiff,targetsub

#function for generating training target data
def GenerateTrainingTarget(targetconcat,trainingPercent):
        TrainingLen = int(math.ceil(len(targetconcat)*(trainingPercent*0.01)))
        t           = targetconcat[:TrainingLen]
        return t
    
#function for generating training input data
def GenerateTrainingDataMatrix(concat_human, TrainingPercent = 80):
        T_len = int(math.ceil(len(concat_human)*0.01*TrainingPercent))
        d2 = concat_human[0:T_len]
        return d2
#function for generating validation input data
def GenerateValData(concat_human, ValPercent, TrainingCount): 
        valSize = int(math.ceil(len(concat_human)*ValPercent*0.01))
        V_End = TrainingCount + valSize
        dataMatrix = concat_human[TrainingCount+1:V_End]
        return dataMatrix
    
#function for generating validation target data
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
        valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
        V_End = TrainingCount + valSize
        t =rawData[TrainingCount+1:V_End]
        return t
#function for calculating variance of input samples with themselves
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
        BigSigma    = np.zeros((len(Data),len(Data)))
        DataT       = np.transpose(Data)
        TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
        varVect     = []
        for i in range(0,len(DataT[0])):
            vct = []
            for j in range(0,int(TrainingLen)):
                vct.append(Data[i][j])    
            varVect.append(np.var(vct))
        #adding variance values to diagonal elements where covariance is zero
        for j in range(len(Data)):
            BigSigma[j][j] = varVect[j]
        if IsSynthetic == True:
            BigSigma = np.dot(3,BigSigma)
        else:
            BigSigma = np.dot(200,BigSigma)
        return BigSigma
    
#function to calculate scalar value of phi(x) for single input data row
def GetScalar(DataRow,MuRow, BigSigInv):  
        R = np.subtract(DataRow,MuRow)
        T = np.dot(BigSigInv,np.transpose(R))  
        L = np.dot(R,T)
        return L
    
#function to return final value of basis function to matrix
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
        phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
        return phi_x
    
#function to generate design matrix of basis functions
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
        TrainingLen = math.ceil(len(Data)*(TrainingPercent*0.01))         
        PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
        BigSigInv = np.linalg.pinv(BigSigma)
        for  C in range(0,len(MuMatrix)):
            for R in range(0,int(TrainingLen)):
                #inputting values of design matrix using gaussian basis function
                PHI[R][C] = GetRadialBasisOut(Data[R], MuMatrix[C], BigSigInv)
        return PHI
    
#function for calculating projected targets by using basis function row wise with subsequent weights
def GetValTest(VAL_PHI,W):
        Y = np.dot(W,np.transpose(VAL_PHI))
        return Y
    
#function for calculating root mean square by evaluating target - projected values   
def GetErms(VAL_TEST_OUT,ValDataAct):
        sum = 0.0
        accuracy = 0.0
        counter = 0
        for i in range (0,len(VAL_TEST_OUT)):
            sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
            if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
                counter+=1
        accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
        return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

#function to calculate linear regression for selected input data and target dataset
def Linear_Regression(inputdata,target,type):
    
    TrainingTarget = np.array(GenerateTrainingTarget(target,trainingpercent))
    TrainingData   = GenerateTrainingDataMatrix(inputdata,trainingpercent)
    
    ValDataAct = np.array(GenerateValTargetVector(target,ValidationPercent, (len(TrainingTarget))))
    ValData    = GenerateValData(inputdata,ValidationPercent, (len(TrainingData)))
    
    TestDataAct = np.array(GenerateValTargetVector(target,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(inputdata,TestPercent, (len(TrainingData)+len(ValData)))
    
    #finding centroids based on number of basis functions
    kmeans = KMeans(n_clusters=M, random_state=0).fit((TrainingData))
    Mu = kmeans.cluster_centers_
    BigSigma= GenerateBigSigma(np.transpose(inputdata), Mu, trainingpercent,IsSynthetic)
    training_phi = GetPhiMatrix(inputdata, Mu, BigSigma, trainingpercent)
    test_phi= GetPhiMatrix(TestData, Mu, BigSigma, 100) 
    val_phi= GetPhiMatrix(ValData, Mu, BigSigma, 100)
    
    W=np.zeros(len(Mu))
    W = np.transpose(W)
    TR_TEST_OUT  = GetValTest(training_phi,W)
    VAL_TEST_OUT = GetValTest(val_phi,W)
    TEST_OUT = GetValTest(test_phi,W)
    W_Now        = np.add(220, W)
    La           = 2
    learningRate = 0.01
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    
    for i in range(0,1263):
            Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(W_Now,training_phi[i])),training_phi[i])
            La_Delta_E_W  = np.dot(La,W_Now)
            Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
            Delta_W       = -np.dot(learningRate,Delta_E)
            W_T_Next      = W_Now + Delta_W
            W_Now         = W_T_Next
            
            #calculating ERMS value for training dataset
            TR_TEST_OUT   = GetValTest(training_phi,W_T_Next) 
            Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
            L_Erms_TR.append(float(Erms_TR.split(',')[1]))
            
            #calculating ERMS value for training dataset
            VAL_TEST_OUT  = GetValTest(val_phi,W_T_Next) 
            Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
            L_Erms_Val.append(float(Erms_Val.split(',')[1]))
            
            #calculating ERMS value for training dataset
            TEST_OUT      = GetValTest(test_phi,W_T_Next) 
            Erms_Test = GetErms(TEST_OUT,TestDataAct)
            L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    
    print ('----------Gradient Descent Solution for',type)
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

'''****** Linear Regression implementation*********'''

"""CONCAT ON HUMAN DATA"""
#calculating datasets for human concat data
concat_human,targetconcat=concat(samepairs,diffpairs,imgfeatures,790,1)
concat_human=np.asarray(concat_human)
targetconcat=np.asarray(targetconcat)
#running linear regression for concat dataset
Linear_Regression(concat_human,targetconcat,'HUMAN CONCAT')

"""SUBTRACTION ON HUMAN DATA"""
#calculating input and target datasets for human subtraction data
concat_humandiff,targetsub=subtract(samepairs,diffpairs,imgfeatures,790,1)
concat_humandiff=np.asarray(concat_humandiff)
targetsub=np.asarray(targetsub)
#running linear regression for concat subtraction dataset
Linear_Regression(concat_humandiff,targetsub,'HUMAN SUBTRACT')

"""CONCAT for GSC DATA"""
#calculating input and target datasets for gsc data files
concat_gsc,target_gsc=concat(samegsc,diffgsc,imggsc,8000,0)
row,col=concat_gsc.shape
concat_gsc=np.asarray(concat_gsc)
target_gsc=np.asarray(target_gsc)
#running linear regression for gsc concat dataset
Linear_Regression(concat_gsc,target_gsc,'GSC CONCAT')

"""SUBTRACTION for GSC DATA"""
#calculating input and target datasets for gsc data files
sub_gsc,subtarget_gsc=subtract(samegsc,diffgsc,imggsc,8000,0)
sub_gsc=np.asarray(sub_gsc)
subtarget_gsc=np.asarray(subtarget_gsc)
#running linear regression for gsc subtraction dataset
Linear_Regression(sub_gsc,subtarget_gsc,'GSC SUBTRACT')

"""******** Logistic Regression Implementation *********"""

def GetScalarl(DataRow,MuRow):  
        R = np.subtract(DataRow,MuRow)
        T = np.transpose(R)
        L = math.sqrt((np.dot(R,T)))
        return L
    
def GetBasisOut(DataRow,MuRow):    
        phi_x = GetScalarl(DataRow,MuRow)
        return phi_x
    
def GetPhiMatrixl(Data, MuMatrix, BigSigma, TrainingPercent = 80):
        DataT = np.transpose(Data)
        TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
        PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
        for  C in range(0,len(MuMatrix)):
            for R in range(0,int(TrainingLen)):
                PHI[R][C] = GetBasisOut(DataT[R], MuMatrix[C])
        return PHI

#defining the hypothesis function for logistic regression which calculates the projected targets
def GetValTestl(VAL_PHI,W):
        s=MinMaxScaler().fit(VAL_PHI[:])
        VAL_PHI[:]=s.transform(VAL_PHI[:])
        Y = 1/(1+np.exp(-np.dot(W,np.transpose(VAL_PHI))))
        return Y
#function defining the cost function for logistic regression    
def GetLoss(VAL_TEST_OUT,ValDataAct):
        sum = 0.0
        accuracy = 0.0
        counter = 0
        for i in range (0,len(VAL_TEST_OUT)):
            #formula of cost function of logistic regression
            sum = sum - ((1-ValDataAct[i])*math.log( 1- VAL_TEST_OUT[i])+ValDataAct[i]*math.log(VAL_TEST_OUT[i]))
            if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
                counter+=1
        accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
        return (str(accuracy) + ',' +  str(sum/len(VAL_TEST_OUT)))

#function for calculating accuracy using logistic regression model
def logistic_regression(inputdata,target,type):
    TrainingTarget = np.array(GenerateTrainingTarget(target,trainingpercent))
    TrainingData   = GenerateTrainingDataMatrix(inputdata,trainingpercent)
    
    ValDataAct = np.array(GenerateValTargetVector(target,ValidationPercent, (len(TrainingTarget))))
    ValData    = GenerateValData(inputdata,ValidationPercent, (len(TrainingData)))
    
    TestDataAct = np.array(GenerateValTargetVector(target,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(inputdata,TestPercent, (len(TrainingData)+len(ValData)))
    
    kmeans = KMeans(n_clusters=M, random_state=0).fit((TrainingData))
    Mu = kmeans.cluster_centers_
    BigSigma= GenerateBigSigma(np.transpose(inputdata), Mu, trainingpercent,IsSynthetic)
    training_phi = GetPhiMatrixl(np.transpose(inputdata), Mu, BigSigma, trainingpercent)
    test_phi= GetPhiMatrixl(np.transpose(TestData), Mu, BigSigma, 100) 
    val_phi= GetPhiMatrixl(np.transpose(ValData), Mu, BigSigma, 100)

    W=np.zeros(len(Mu))
    W = np.transpose(W)
    W_Now        = np.add(0, W)
    TR_TEST_OUT  = GetValTestl(training_phi,W)
    VAL_TEST_OUT = GetValTestl(val_phi,W)
    TEST_OUT = GetValTestl(test_phi,W)
    La           = 2
    learningRate = 0.01
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
        
    for i in range(0,1263):
        Delta_E_D     = -np.dot((TrainingTarget[i] - 1/(1+np.exp(np.dot(W_Now,training_phi[i])))),training_phi[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
        
        TR_TEST_OUT   = GetValTestl(training_phi,W_T_Next) 
        Erms_TR = GetLoss(TR_TEST_OUT,TrainingTarget)
        L_Erms_TR.append([float(Erms_TR.split(',')[0]),float(Erms_TR.split(',')[1])])
        
        VAL_TEST_OUT  = GetValTestl(val_phi,W_T_Next) 
        Erms_Val      = GetLoss(VAL_TEST_OUT,ValDataAct)
        L_Erms_Val.append([float(Erms_Val.split(',')[0]),float(Erms_Val.split(',')[1])])
        
        TEST_OUT      = GetValTestl(test_phi,W_T_Next) 
        Erms_Test = GetLoss(TEST_OUT,TestDataAct)
        L_Erms_Test.append([float(Erms_Test.split(',')[0]),float(Erms_Test.split(',')[1])])

    L_Erms_TR = np.array(L_Erms_TR)
    L_Erms_Val = np.array(L_Erms_Val)
    L_Erms_Test = np.array(L_Erms_Test)
    print ('----------Gradient Descent Solution for',type)
    #function used to calculate the accuracy for specific dataset
    def accuracy(L_Erms_Test):
        minimum=0
        temp=min(L_Erms_Test[:,1])
        row,col=np.shape(L_Erms_Test)
        for i in range(row):
            if (L_Erms_Test[i,1]==temp):
                minimum=L_Erms_Test[i,0]
        return minimum
    #printing accuracy measures for training,testing and validation datasets
    print("Accuracy training = ",accuracy(L_Erms_TR))
    print("Accuracy validation = ",accuracy(L_Erms_Val))
    print("Accuracy test = ",accuracy(L_Erms_Test))

#running logistic regression for human dataset using concatenation of features
concat_human,targetconcat=concat(samepairs,diffpairs,imgfeatures,790,1)
concat_human=np.asarray(concat_human)
targetconcat=np.asarray(targetconcat)
logistic_regression(concat_human,targetconcat,'Logistic Human Concat')

#running logistic regression for human dataset using subtraction of features
concat_humandiff,targetsub=subtract(samepairs,diffpairs,imgfeatures,790,1)
concat_humandiff=np.asarray(concat_humandiff)
targetsub=np.asarray(targetsub)
logistic_regression(concat_humandiff,targetsub,'Logistic Human Subtract')

#running logistic regression for gsc dataset using concatenation of features
concat_gsc,target_gsc=concat(samegsc,diffgsc,imggsc,8000,0)
row,col=concat_gsc.shape
concat_gsc=np.asarray(concat_gsc)
target_gsc=np.asarray(target_gsc)
logistic_regression(concat_gsc,target_gsc,'Logistic GSC Concat')

#running logistic regression for gsc dataset using subtraction of features
sub_gsc,subtarget_gsc=subtract(samegsc,diffgsc,imggsc,8000,0)
sub_gsc=np.asarray(sub_gsc)
subtarget_gsc=np.asarray(subtarget_gsc)
logistic_regression(sub_gsc,subtarget_gsc,'Logistic GSC Subtract')

'''******** Neural Network Implementation *********'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
'''Human data Concat'''

input_size = 18      
drop_out = 0.4
first_dense_layer_nodes  = 1581 
second_dense_layer_nodes = 790
third_dense_layer_nodes = 2

def get_model():

    model = Sequential() #creating a sequential model with linear layers
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))#adding dense layer with input size parameter to make model understand input shape
    model.add(Activation('relu')) #adding activation function 'relu' to first hidden layer
    
    model.add(Dropout(drop_out))
    #adding second hidden layer for sequential model
    model.add(Dense(second_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax')) #adding activation function 'softmax' to output layer
    # Softmax To identify outputs for a multi class classification problem statement
    
    model.summary()
    
    model.compile(optimizer='rmsprop',             
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])           
    
    return model  

model = get_model()

validation_data_split = 0.2
num_epochs =5000      
model_batch_size = 128      
tb_batch_size = 32
early_patience = 100      

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min') #setting for defining early-stopping of epochs during training of model

concat_human,targetconcat=concat(samepairs,diffpairs,imgfeatures,790,1)
#coverting target dataset into two classes using categorical function
targetconcat=np_utils.to_categorical(np.array(targetconcat),2)
history = model.fit(concat_human
                    , targetconcat
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                    , shuffle=True
                   )

df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))

'''Human data Subtraction'''
input_size = 9      
drop_out = 0.2
first_dense_layer_nodes  = 1578
second_dense_layer_nodes = 790
third_dense_layer_nodes = 2
def get_model():

    model = Sequential() #creating a sequential model with linear layers
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    #adding dense layer with input size parameter to make model understand input shape
    model.add(Activation('relu')) 
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax')) #adding activation function 'softmax' to output layer
    
    
    model.summary()
    
    model.compile(optimizer='rmsprop',             
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])           
    
    return model  

model = get_model()

validation_data_split = 0.2
num_epochs =5000      
model_batch_size = 128      
tb_batch_size = 32
early_patience = 200     

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
concat_humandiff,targetsub=subtract(samepairs,diffpairs,imgfeatures,790,1)
targetsub.drop(targetsub.tail(3).index,inplace=True)
targetsub=np_utils.to_categorical(np.array(targetsub),2)

history = model.fit(concat_humandiff
                    , targetsub
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                    , shuffle=True
                   )

df2 = pd.DataFrame(history.history)
df2.plot(subplots=True, grid=True, figsize=(10,15))

'''GSC data Concat '''
input_size = 1024      
drop_out = 0.3
first_dense_layer_nodes  = 2500
second_dense_layer_nodes = 1250
third_dense_layer_nodes = 2
def get_model():

    model = Sequential() 
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu')) 
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax')) 
    
    model.summary()
    model.compile(optimizer='sgd',             
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])           
    return model  

model = get_model()

validation_data_split = 0.3
num_epochs =1000     
model_batch_size = 128      
tb_batch_size = 32
early_patience =100     

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
concat_gsc,target_gsc=concat(samegsc,diffgsc,imggsc,3900,0)
target_gsc=np_utils.to_categorical(np.array(target_gsc),2)

history = model.fit(concat_gsc
                    , target_gsc
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                    ,shuffle=True
                   )

df2 = pd.DataFrame(history.history)
df2.plot(subplots=True, grid=True, figsize=(10,15))

'''GSC data Subtraction '''
input_size = 512      
drop_out = 0.3
first_dense_layer_nodes  = 2500
second_dense_layer_nodes = 1250
third_dense_layer_nodes = 2
def get_model():

    model = Sequential() 
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu')) 
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax')) 
    
    model.summary()
    model.compile(optimizer='sgd',             
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])           
    return model  

model = get_model()

validation_data_split = 0.3
num_epochs =1000     
model_batch_size = 128      
tb_batch_size = 32
early_patience =100     

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
sub_gsc,subtarget_gsc=subtract(samegsc,diffgsc,imggsc,3900,0)
subtarget_gsc.drop(subtarget_gsc.tail(1).index,inplace=True)
subtarget_gsc=np_utils.to_categorical(np.array(subtarget_gsc),2)

history = model.fit(sub_gsc
                    , subtarget_gsc
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                    ,shuffle=True
                   )

df2 = pd.DataFrame(history.history)
df2.plot(subplots=True, grid=True, figsize=(10,15))
