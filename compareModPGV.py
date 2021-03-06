import numpy as np
import pdb
import pylab as plt
import pickle
from matplotlib import gridspec

import


f, arr = plt.subplots(4,figsize=(17,12))
#f = plt.figure(figsize=(17,12))
#gs = gridspec.GridSpec(4,2, width_ratios=[4,1])
#arr=[[],[],[],[],[]]
#arr[0] = plt.subplot(gs[0])
#arr[1] = plt.subplot(gs[2])
#arr[2] = plt.subplot(gs[4])
#arr[3] = plt.subplot(gs[6])
#arr[4] = plt.subplot(gs[1])

legendstr = [] 

## plot GM   
for M in range(5):
    for M2 in range(4):
    	#print M, M2
    
        if M2 <=1: #plot Mag
            delta0 = [50,150]
            delta = np.sqrt(delta0[M2]**2 + 8**2)
            delta1 = delta
            mw =  np.arange(3.5,6+0.1,0.1)
            
        if M2 >= 2:  #plot Dist
            delta = np.sqrt(8**2 + np.arange(1,251,1)**2)
            delta1 = np.arange(1,251,1)
            mw0 = [4,5]
            mw = mw0[M2-2]
        ml = (mw - 1.145) / 0.812

	
            
        #Zonen#################################
        #Model = 1, MaxDist = 400, Filt= 99, Mag = [2,5.5], StatCorr = False, Zone = 1)
        #if M >= 6 and M<=10: #WEGINGER 2015 [g]
        #    if M == 6: C = [-5.064, 1.38, -0.0387, -2.21]	# Zone = 0
        #    if M == 7: C = [-7.16, 2.43, -0.166, -2.22] # Zone = 1
        #    if M == 8: C = [-6.85, 2.31, -0.152, -2.26] # Zone = 2
        #    if M == 9: C = [-5.036, 1.28, -0.025, -2.129]# [-4.14, 0.922, 0.0165, -2,20] # Zone = 3
        #    if M == 10: C = [-7.16, 2.42, -0.166, -2.22] # Zone = 4
        #    Y0 = 10**(C[0] + C[1] * mw + C[2] * mw ** 2 + C[3] * np.log10(delta)) *9.81 #[m/s2]
        #    style = '-'; alph = 1
        #    if M2 == 0: legendstr.append('Weginger15 Z%i'%(M-6))
        
        #Model###################################      
        if M == 0:  #WEGINGER 2015 [m/s]
            C = [-5.057, 0.573, 0.074, -1.71408012]	#Model = 1, MaxDist = 300, Filt= 99, Mag = [2,5.5] Zone = 0
            Y0 = 10**(C[0] + C[1] * mw + C[2] * mw ** 2 + C[3] * np.log10(delta)) 
            style = '-'; alph = 1; lwi = 2;
            if M2 == 0: legendstr.append('Weginger15_1')
        if M == 1:  #WEGINGER 2015 [m/s]
            C = [-3.79 ,-0.0556, 0.15, -1.7137]	#Model = 1, MaxDist = 300, Filt= 99, Mag = [2,6] Zone = 0
            Y0 = 10**(C[0] + C[1] * mw + C[2] * mw ** 2 + C[3] * np.log10(delta)) 
            style = '-'; alph = 1; lwi = 2;
            if M2 == 0: legendstr.append('Weginger15_2')  
        if M == 2:  #WEGINGER 2016				#Model = 3, MaxDist = 300, Filt= 99, Mag = [2,5.5] Zone = 0
            C = [-5.79, 0.744, 0.0784,-1.329,-0.09802056]
            Y0 = 10 **(C[0] + C[1] * mw + C[2] * mw ** 2 + ( C[3] + C[4] * mw ) * np.log10 ( delta )) 
            if M2 == 0: legendstr.append('Weginger16')
            style = '-'; alph = 1
        if M == 3:  #WEGINGER 2011
            C = [-6.42, 1.21, -1.714]			#Model = 2, MaxDist = 300, Filt= 99, Mag = [2,6] Zone = 0
            Y0 = 10**(C[0] + C[1] * mw + C[2] * np.log10(delta)) 
            if M2 == 0: legendstr.append('Weginger11')
 
        if M == 4:  #WEGINGER 2015 [m/s]
            C = [-5.87446116,0.94969181,-1.38895154]	#Model = 1, MaxDist = 300, Filt= 99, Mag = [2,5.5] Zone = 0
            Y0 = 10**(C[0] + C[1] * mw + C[2] * np.log10(delta)) 
            #Y0 = 10**(C[0] + C[1] * mw + C[2] * mw ** 2 + C[3] * np.log10(delta)) 
            style = '-'; alph = 1; lwi = 2;
            if M2 == 0: legendstr.append('Weginger15_SN')  
            
            
        #if M == 8:	 #WEGINGER 2015	V2
        #    C = [-6.50 ,1.138, 0.022, -1.438, -0.0644]
        #    #C = [-6.263 ,1.84, -0.0754, -1.94, -0.073]
        #    Y0 = 10 **(C[0] + C[1] * mw + C[2] * mw ** 2 + ( C[3] + C[4] * mw ) * np.log10 ( delta ))

        #Data###########################################
        #if M == 6:  #WEGINGER 2015 [g]
        #   #C = [-5.065 ,1.383, -0.038, -2.209]
        #    C = [-5.72 ,1.72, -0.08, -2.229]	#Model = 1, MaxDist = 350, Filt= 99, Mag = [2,5.5] Zone = 0
        #    Y0 = 10**(C[0] + C[1] * mw + C[2] * mw ** 2 + C[3] * np.log10(delta)) *9.81
        #    style = '-'; alph = 1
        #    if M2 == 0: legendstr.append('Weginger15') 
        #
        #if M == 7:  #WEGINGER 15 HH
        #    C = [-5.63, 1.62, -0.0606, -2.234]
        #    Y0 = 10**(C[0] + C[1] * mw + C[2] * mw ** 2 + C[3] * np.log10(delta)) *9.81
        #    style = '-' 
        #    if M2 == 0: legendstr.append('Weginger15 HH')
        #    
        #if M == 8:  #WEGINGER 15 HN, HL
        #    C = [-4.46, 1.236, -0.042, -2.144]  #odel = 1, MaxDist = 350, Filt= 99, Mag = [2,5.5], Zone = 0
        #    Y0 = 10**(C[0] + C[1] * mw + C[2] * mw ** 2 + C[3] * np.log10(delta)) *9.81
        #    style = '-' 
        #    if M2 == 0: legendstr.append('Weginger15 HN HL')  
        

              
    
        if M2 <= 1: #plot Mag
            #gs = gridspec.GridSpec(3,2)  
            #pdb.set_trace()    
            arr[M2].semilogy(mw, Y0,style,alpha = alph,lw = lwi)
            
        if M2 >= 2:  #plot Dist
            #plt.subplot(gs[1,0])
            #pdb.set_trace()
            arr[M2].loglog(delta1, Y0,style,alpha = alph,lw = lwi)

##plot OQ GM Dist
file = open('./GMPE_oq_dist.txt'); 
dict = pickle.load(file)

class cloqGM: pass
oqGM = cloqGM(); oqGM.list = dict[0]; oqGM.imts = dict[1]; oqGM.params = dict[2]
oqGM.magnitudes = dict[3]; oqGM.distances = dict[4]; oqGM.val = dict[5]

style = '--'; alph = 1
for curGM in oqGM.list:
    if len(oqGM.val[curGM]['PGV']):
        arr[2].loglog(oqGM.distances['repi'], oqGM.val[curGM]['PGV'][0]/100, style,alpha = alph)	#Mw4
        arr[3].loglog(oqGM.distances['repi'], oqGM.val[curGM]['PGV'][1]/100, style,alpha = alph)	#Mw5
        legendstr.append(curGM)

##plot OQ GM Mag
file = open('./GMPE_oq_mag.txt'); 
dict = pickle.load(file)

class cloqGM: pass
oqGM = cloqGM(); oqGM.list = dict[0]; oqGM.imts = dict[1]; oqGM.params = dict[2]
oqGM.magnitudes = dict[3]; oqGM.distances = dict[4]; oqGM.val = dict[5]

style = '--'; alph = 1
for curGM in oqGM.list:
    if len(oqGM.val[curGM]['PGV']):  
        arr[1].semilogy(oqGM.magnitudes, oqGM.val[curGM]['PGV'][:,2]/100 ,style,alpha = alph)		#150km
        arr[0].semilogy(oqGM.magnitudes, oqGM.val[curGM]['PGV'][:,0]/100 ,style,alpha = alph)		#50km
        legendstr.append(curGM)


    

## show all
arr[0].set_xlabel('Mw'); arr[0].set_ylabel('PGV [m/s2] in 50km')
arr[1].set_xlabel('Mw'); arr[1].set_ylabel('PGV [m/s2] in 150km') 
arr[2].set_xlabel('epi dist [km]');  arr[2].set_ylabel('PGV [m/s2] Mw = 4')
arr[3].set_xlabel('epi dist [km]');  arr[3].set_ylabel('PGV [m/s2] Mw = 5')
arr[3].legend(legendstr, bbox_to_anchor=(1.1,3))

arr[0].set_xlim([3.5,6.25]);arr[1].set_xlim([3.5,6.25]);arr[2].set_xlim([1,500]);arr[3].set_xlim([1,500]);

plt.savefig('/home/weginger/stefan/curFig', dpi=250)       
plt.show()
