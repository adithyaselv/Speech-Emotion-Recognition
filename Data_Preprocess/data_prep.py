#Author : Adithya Selvaprithiviraj
#Date : 14/03/2017
#Description : Data preprocessing for training

import os
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import numpy as np

all_files=os.listdir("./wav")
ohm_school=[]

label_file=open("./labels/IS2009EmotionChallenge/chunk_labels_5cl_corpus.txt",'r')

label_lines=label_file.readlines()
label_lines=[line.strip().split(" ") for line in label_lines]


label_dict={}

for label_line in label_lines:
    label_dict[label_line[0]]=label_line[1]

label_map={"A":[1,0,0,0,0],"E":[0,1,0,0,0],"N":[0,0,1,0,0],"P":[0,0,0,1,0],"R":[0,0,0,0,1]}
        
# print label_dict
label_file.close()
# print len(ohm_school)

def makeFeatureVector(fileName):
    (rate,sig)=wav.read("./wav/"+fileName)
    mfcc_feat=mfcc(sig,rate,appendEnergy=True)
    mfcc_feat=mfcc_feat[:100]
    delta1=delta(mfcc_feat,5)
    delta2=delta(delta1,5)
  
    mfcc_feat=np.array(mfcc_feat)
    mfcc_feat=mfcc_feat.ravel()
    delta1=np.array(delta1)
    delta1=delta1.ravel()
    delta2=np.array(delta2)
    delta2=delta2.ravel()
    lab_vec=label_map[label_dict[fileName[:-4]]]
    lab_vec=np.array(lab_vec)

    all_features=np.concatenate([mfcc_feat,delta1,delta2,lab_vec])

    # print all_features.shape
    return all_features


def writeIntoFile(all_files):
    flag=0
    full=np.array([])
    lolol=0
    for i in all_files:
        if "Ohm" in i:
            if not flag:
                print "here"
                flag=1
                full=np.array([makeFeatureVector(i)])
            else:
                try:
                    full=np.append(full,np.array([makeFeatureVector(i)]),axis=0)
                except Exception as e:
                    print e
            lolol +=1
            print lolol
            
            # if lolol==150:
            #     break

    np.savetxt("./csv/ohm_all.csv",full,delimiter=",")

def writeMultipleFiles(all_files):
    anger=np.empty((0,3905), float)
    emphatic=np.empty((0,3905), float)
    neutral=np.empty((0,3905), float)
    joyful=np.empty((0,3905), float)
    rest=np.empty((0,3905), float)

    lolol=0
    for i in all_files:
        if "Ohm" in i:
            if label_dict[i[:-4]]=="A":
                try:
                    anger=np.append(anger,np.array([makeFeatureVector(i)]),axis=0)
                except Exception as e:
                    print e
            elif label_dict[i[:-4]]=="E":
                try:
                    emphatic=np.append(emphatic,np.array([makeFeatureVector(i)]),axis=0)
                except Exception as e:
                    print e
            elif label_dict[i[:-4]]=="N":
                try:
                    neutral=np.append(neutral,np.array([makeFeatureVector(i)]),axis=0)
                except Exception as e:
                    print e
            elif label_dict[i[:-4]]=="P":
                try:
                    joyful=np.append(joyful,np.array([makeFeatureVector(i)]),axis=0)
                except Exception as e:
                    print e
            elif label_dict[i[:-4]]=="R":
                try:
                    rest=np.append(rest,np.array([makeFeatureVector(i)]),axis=0)
                except Exception as e:
                    print e
            lolol +=1
            print lolol

            # if lolol == 6000:
            #     break
          
    print "DEBUG : "
    print anger.shape
    print emphatic.shape
    print neutral.shape
    print joyful.shape
    print rest.shape


    np.savetxt("./csv/ohm_anger.csv",anger,delimiter=",")
    np.savetxt("./csv/ohm_emphatic.csv",emphatic,delimiter=",")
    np.savetxt("./csv/ohm_neutral.csv",neutral,delimiter=",")
    np.savetxt("./csv/ohm_joyful.csv",joyful,delimiter=",")
    np.savetxt("./csv/ohm_rest.csv",rest,delimiter=",")



writeMultipleFiles(all_files)


            





