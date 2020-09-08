#!/usr/bin/env python
import sys,os,math
from sklearn import svm
from scipy import interpolate
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def sigmoid(x):
    return 1/(1+np.exp(-x))

def getCDF(list):
    x=[0]
    y=[0]
    print("[getCDF]Nums of values: %d"%(len(list)))
    hist,bin_edges=np.histogram(list,normed=False,density=True)
    x.extend(bin_edges[:len(bin_edges)-1])
    x.append(1)
    
    for i in range(1,len(hist)+1):
        y.append(sum(hist[:i]*np.diff(bin_edges)[:i]))
    y.append(1)
    print("Range:%f -> %f"%(x[0],x[len(x)-1]))
    return interpolate.interp1d(x,y,kind="quadratic")

# Compute Confidence as w
def computeC(s,f1,f2):
    return np.abs(f1(s)-(1-f2(s)))

def getWvector(x,gamma):
    w=[]
    for i in range(0,len(x)):
        w.append(computeC(x[i],gamma[0][i],gamma[1][i]))
    return w
    
def load_data(data_path,n):
    list=[]
    print("Load data from "+data_path+"...")
    with open(data_path,'r') as f:
        content=f.readlines()
        for line in content:
            line=line.strip()
            data_list=line.split()
            if(n!=len(data_list)):
                print('[error] the %s file has no %s fields'%(data_path,n))
                exit(1)
            if not data_list[0].startswith("#"):
                list.append(data_list)
    return list

#### main #####
options={
"write_weight":"",
"normalize":True,
"confidence":False,
"svm": False,
"lda": False,
"label_column":"1"}

n=1
for i in range(1,len(sys.argv)):
    if sys.argv[i].startswith('--'): 
        parameter = sys.argv[i][2:].split("=")
        if(parameter[1]=="true"):
            options[parameter[0].replace("-","_")]=True
        elif(parameter[1]=="false"):
            options[parameter[0].replace("-","_")]=False
        elif(parameter[1]!=""):
            options[parameter[0].replace("-","_")]=parameter[1]
        n+=1
if len(sys.argv)-n != 3 :
    print('usage: '+sys.argv[0]+' [--write-weight="" | file-path ] <trials> <score-scp> <out-score>')
    print('e.g.: '+sys.argv[0]+' --write-weight=test_1s/fusion.weight test_1s/trials test_1s/score.scp test_1s/fusion.score')
    exit(1)

print(options)
print("begin")

trials_file=sys.argv[n]
scp_file=sys.argv[n+1]
out_file=sys.argv[n+2]

# load trials and score
trials=load_data(trials_file,3)
# scp with `score, filepath` list
scp=load_data(scp_file,2)

trials_dict={}
for i in range(0,len(trials)):
    if options["label_column"] == "3":
        trials_dict[trials[i][0]+" "+trials[i][1]]=trials[i][2]
    elif options["label_column"] == '1':
        trials_dict[trials[i][1]+" "+trials[i][2]]=trials[i][0]
    else:
        raise ValueError("not support")
    
score=[]
for i in range(0,len(scp)):
    dict={}
    temp = load_data(scp[i][1], 3)
    for j in range(0, len(temp)):
        if options["label_column"] == "3":
            dict[temp[j][0]+" "+temp[j][1]]=float(temp[j][2]) if not options["normalize"] else sigmoid(float(temp[j][2]))
        elif options["label_column"] == '1':
            dict[temp[j][1]+" "+temp[j][2]]=float(temp[j][0]) if not options["normalize"] else sigmoid(float(temp[j][0]))
        else:
            raise ValueError("not support")
    score.append(dict)

# x for sys1,sys2,...,sysn score
# y be one of non-target, target
x=[]
y=[]
w=[]
print("Transform data to vector...")
for i in range(0,len(trials)):
    if options["label_column"] == "3":
        key = trials[i][0] + " " + trials[i][1]
    elif options["label_column"] == "1":
        key = trials[i][1] + " " + trials[i][2]
    else:
        raise ValueError("not support")
    temp=[]
    for j in range(0,len(score)):
        temp.append(score[j][key])
    x.append(temp)
    if(trials_dict[key]=="target" or trials_dict[key]=='1'):
        y.append(1)
    else:
        y.append(-1)
        
# train model
if(options["confidence"]==True):
    print("Prapare data for CDF computation ...")
    target=[]
    nontarget=[]
    for i in range(0,len(score)):
        target.append([])
        nontarget.append([])
        
    for i in range(0,len(trials)):
        if options["label_column"] == "3":
            key = trials[i][0] + " " + trials[i][1]
        elif options["label_column"] == "1":
            key = trials[i][1] + " " + trials[i][2]
        else:
            raise ValueError("not support")

        for j in range(0,len(score)):
            if(trials_dict[key]=="target" or trials_dict[key]=='1'):
                target[j].append(score[j][key])
            else:
                nontarget[j].append(score[j][key])

    print("Compute gamma for confidence...")
    gamma=[[],[]] # index-0 -> target,index-1 -> nontarget
    for i in range(0,len(score)):
        gamma[0].append(getCDF(target[i]))
        gamma[1].append(getCDF(nontarget[i]))
    print("Computation done.")
        
else:   
    model = None
    if options["lda"]==True:
        print("Train Lda model...(it needs some time)...")  
        model = LinearDiscriminantAnalysis(solver='svd', n_components=1)
    elif options["svm"]==True:
        print("Train svm model...(it needs some time)...")  
        model = svm.SVC(kernel='linear', max_iter=-1,C=1,random_state= 777)
    model.fit(x,y)
    print("Training done.")
    w=model.coef_[0]
    b=model.intercept_[0]

    if(options["write_weight"]!=""):
        print("write weight to %s..."%(options["write_weight"]))
        txt_w=w.tolist()
        #txt_b=0
        txt_b=b
        file=open(options["write_weight"],"w+")
        file.write("[ ")
        for i in range(0,len(txt_w)):
            file.write("%f "%(txt_w[i]))
        file.write("%f ]\n"%(txt_b))
        file.close()
        
    print("weight as follows:")
    print("w =",w,"\nb =",b)
    print("\n")

# apply transform to raw score
print("Write fusion score to %s..."%(out_file))
f=open(out_file,"w+")

for i in range(0,len(trials)):
    if(options["confidence"]==True):
        w=getWvector(x[i],gamma)
        b=0
    value=np.dot(x[i],w)+b
    if options["label_column"] == "3":
        f.write("%s %s %f\n"%(trials[i][0],trials[i][1],value))
    elif options["label_column"] == "1":
        f.write("%d %s %s\n"%(value, trials[i][1],trials[i][2]))
    else:
        raise ValueError("not support")
f.close()

print("All done.")
