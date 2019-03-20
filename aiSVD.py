#siddharth Vadgama
#UTA ID-1001397508
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

A=genfromtxt('digitsTrainCombined.csv', delimiter=',')
z=genfromtxt('digitsTestCombined.csv', delimiter=',')
A_0_digits=A[:256,:400]

A_1_digits=A[:256,400:800]
A_2_digits=A[:256,800:1200]
A_3_digits=A[:256,1200:1600]
A_4_digits=A[:256,1600:2000]
A_5_digits=A[:256,2000:2400]
A_6_digits=A[:256,2400:2800]
A_7_digits=A[:256,2800:3200]
A_8_digits=A[:256,3200:3600]
A_9_digits=A[:256,3600:4000]

z_0_digits=A[:256,:100]
z_1_digits=A[:256,100:200]
z_2_digits=A[:256,200:300]
z_3_digits=A[:256,300:400]
z_4_digits=A[:256,400:500]
z_5_digits=A[:256,500:600]
z_6_digits=A[:256,600:700]
z_7_digits=A[:256,700:800]
z_8_digits=A[:256,800:900]
z_9_digits=A[:256,900:1000]

U_0,S_0,Vt_0=np.linalg.svd(A_0_digits, full_matrices=False)
U_1,S_1,Vt_1=np.linalg.svd(A_1_digits, full_matrices=False)
U_2,S_2,Vt_2=np.linalg.svd(A_2_digits, full_matrices=False)
U_3,S_3,Vt_3=np.linalg.svd(A_3_digits, full_matrices=False)
U_4,S_4,Vt_4=np.linalg.svd(A_4_digits, full_matrices=False)
U_5,S_5,Vt_5=np.linalg.svd(A_5_digits, full_matrices=False)
U_6,S_6,Vt_6=np.linalg.svd(A_6_digits, full_matrices=False)
U_7,S_7,Vt_7=np.linalg.svd(A_7_digits, full_matrices=False)
U_8,S_8,Vt_8=np.linalg.svd(A_8_digits, full_matrices=False)
U_9,S_9,Vt_9=np.linalg.svd(A_9_digits, full_matrices=False)

#for K=1
k=[1,5,20,100,256]
for ix in k:
	print('--------------------')
	I=np.identity(256)
	U_0_k=U_0[:,:ix]
	U_0_kT=np.transpose(U_0_k)
	U_temp_0=np.matmul(U_0_k,U_0_kT)
	temp_0=I-U_temp_0

	U_1_k=U_1[:,:ix]
	U_1_kT=np.transpose(U_1_k)
	U_temp_1=np.matmul(U_1_k,U_1_kT)
	temp_1=I-U_temp_1

	U_2_k=U_2[:,:ix]
	U_2_kT=np.transpose(U_2_k)
	U_temp_2=np.matmul(U_2_k,U_2_kT)
	temp_2=I-U_temp_2

	U_3_k=U_3[:,:ix]
	U_3_kT=np.transpose(U_3_k)
	U_temp_3=np.matmul(U_3_k,U_3_kT)
	temp_3=I-U_temp_3

	U_4_k=U_4[:,:ix]
	U_4_kT=np.transpose(U_4_k)
	U_temp_4=np.matmul(U_4_k,U_4_kT)
	temp_4=I-U_temp_4

	U_5_k=U_5[:,:ix]
	U_5_kT=np.transpose(U_5_k)
	U_temp_5=np.matmul(U_5_k,U_5_kT)
	temp_5=I-U_temp_5

	U_6_k=U_6[:,:ix]
	U_6_kT=np.transpose(U_6_k)
	U_temp_6=np.matmul(U_6_k,U_6_kT)
	temp_6=I-U_temp_6

	U_7_k=U_7[:,:ix]
	U_7_kT=np.transpose(U_7_k)
	U_temp_7=np.matmul(U_7_k,U_7_kT)
	temp_7=I-U_temp_7

	U_8_k=U_8[:,:ix]
	U_8_kT=np.transpose(U_8_k)
	U_temp_8=np.matmul(U_8_k,U_8_kT)
	temp_8=I-U_temp_8

	U_9_k=U_9[:,:ix]
	U_9_kT=np.transpose(U_9_k)
	U_temp_9=np.matmul(U_9_k,U_9_kT)
	temp_9=I-U_temp_9
	right=[0]*10
	wrong=[0]*10
	total=0
	norms=[None]*10
	for i in range(0,1000):
		t_0=np.matmul(temp_0,z[:,i])
		t_1=np.matmul(temp_1,z[:,i])
		t_2=np.matmul(temp_2,z[:,i])
		t_3=np.matmul(temp_3,z[:,i])
		t_4=np.matmul(temp_4,z[:,i])
		t_5=np.matmul(temp_5,z[:,i])
		t_6=np.matmul(temp_6,z[:,i])
		t_7=np.matmul(temp_7,z[:,i])
		t_8=np.matmul(temp_8,z[:,i])
		t_9=np.matmul(temp_9,z[:,i])
	
		norms[0]=np.linalg.norm(t_0,2)
		norms[1]=np.linalg.norm(t_1,2)
		norms[2]=np.linalg.norm(t_2,2)
		norms[3]=np.linalg.norm(t_3,2)
		norms[4]=np.linalg.norm(t_4,2)
		norms[5]=np.linalg.norm(t_5,2)
		norms[6]=np.linalg.norm(t_6,2)
		norms[7]=np.linalg.norm(t_7,2)
		norms[8]=np.linalg.norm(t_8,2)
		norms[9]=np.linalg.norm(t_9,2)
	
		
		j=i/100
		j=int(j)
		total=total+1
		if norms.index(min(norms))==j:
			right[j]=right[j]+1
		else: 
			wrong[j]=wrong[j]+1
	total_r=sum(right)
	total_w=sum(wrong)
	per=total_r/total*100
	print('for value of K =',ix)
	for x in range(0,9):
		print('digit =',x,'right =',right[x],'wrong =',wrong[x])
	print('total = ',total,'total right=',total_r,'total wrong=',total_w,'accuracy=',round(per,2))
