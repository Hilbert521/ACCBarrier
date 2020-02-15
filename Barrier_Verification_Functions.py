#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:05:22 2019

@author: maxencedutreix
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from math import exp
from math import sqrt
from math import pi
from scipy.special import erf
from scipy.integrate import quad
from scipy.stats import norm
from matplotlib import rc
import timeit
import bisect
import sys
import igraph
import scipy.sparse as sparse
import scipy.sparse.csgraph
import matlab.engine
import itertools
import StringIO
import pickle
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import os


sys.setrecursionlimit(10000)




LOW_1 = 0.0
UP_1 = 4.0
LOW_2 = 0.0
UP_2 = 4.0
sigma1 = sqrt(0.1)
sigma2 = sqrt(0.1)
mu1 = 0.0
mu2 = 0.0
Gaussian_Width_1 = 0.2
Gaussian_Width_2 = 0.2
Semi_Width_1 = Gaussian_Width_1/2.0
Semi_Width_2 = Gaussian_Width_2/2.0

Time_Step = 0.05








def Probability_Interval_Computation_Barrier(Target_Set, Domain, Reachable_States, start_index):
    
       
    Lower = np.array(np.zeros((Target_Set.shape[0],Target_Set.shape[0])))
    Upper = np.array(np.zeros((Target_Set.shape[0],Target_Set.shape[0])))
    Pre_States = [[] for x in range(Target_Set.shape[0])]
    Is_Bridge_State = np.zeros(Target_Set.shape[0])
    Bridge_Transitions = [[] for x in range(Target_Set.shape[0])]  
    Known_Bounds = [[[0.0, 1.0] for x in range(Target_Set.shape[0])] for y in range(Target_Set.shape[0])]
    
      
    eng = matlab.engine.start_matlab() #Start Matlab Engine
      
    for j in range(Target_Set.shape[0]):
        for h in range(len(Reachable_States[j])):
            
            out = StringIO.StringIO()
            err = StringIO.StringIO()
                        
            Res = eng.Bounds_Computation_Verification(matlab.double(list(itertools.chain.from_iterable(Target_Set[j].tolist()))), matlab.double(list(itertools.chain.from_iterable(Target_Set[Reachable_States[j][h]].tolist()))), matlab.double(Known_Bounds[j][Reachable_States[j][h]]), matlab.double(Domain), stdout=out,stderr=err)


            H = Res[0][0]
            L = Res[0][1]
            Known_Bounds[j][Reachable_States[j][h]][0] = Res[0][2]
            Known_Bounds[j][Reachable_States[j][h]][1] = Res[0][3]
            if H > 0:
                if L == 0:
                    Is_Bridge_State[j] = 1
                    Bridge_Transitions[j].append(Reachable_States[j][h])
            else:
                Reachable_States[j].remove(Reachable_States[j][h])
                    
            Lower[j][Reachable_States[j][h]] = L
            Upper[j][Reachable_States[j][h]] = H

            

    eng.quit()
    

           
    return (Lower,Upper, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States, Known_Bounds)


    

def Initial_Partition_Plot(Space):
    
    #Plots the initial state space before verification/synthesis/refinement
       
    fig = plt.figure('Partition P')
    plt.title(r'Initial Partition P', fontsize=25)
        
    plt.plot([0, 4], [0, 0], color = 'k')
    plt.plot([0, 4], [1.0,1.0], color = 'k')
    plt.plot([0, 4], [2.0,2.0], color = 'k')
    plt.plot([0, 4], [3.0,3.0], color = 'k')
    plt.plot([0, 4], [4,4], color = 'k')
    plt.plot([0, 0], [0 ,4], color = 'k')
    plt.plot([1.0, 1.0], [0,4], color = 'k')
    plt.plot([2.0, 2.0], [0,4], color = 'k')
    plt.plot([3.0, 3.0], [0,4], color = 'k')
    plt.plot([4.0, 4.0], [0,4], color = 'k')
    
    ax = plt.gca()
    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    ax.set_xlabel('x1', fontsize=20)
    ax.set_ylabel('x2', fontsize=20)
    plt.savefig('Partition.pdf', bbox_inches='tight')
    
    return 1    



    
def Build_Product_IMC(T_l, T_u, A, L, Acc, Reachable_States, Is_Bridge_State, Bridge_Transitions):
    
      
    
    IA_l = np.zeros((T_l.shape[0]*len(A),T_l.shape[0]*len(A)))
    IA_u = np.zeros((T_l.shape[0]*len(A),T_l.shape[0]*len(A)))
    Is_A = np.zeros(T_l.shape[0]*len(A))
    Is_N_A = np.zeros(T_l.shape[0]*len(A))
    Which_A = [[] for x in range(T_l.shape[0]*len(A))]
    Which_N_A = [[] for x in range(T_l.shape[0]*len(A))]
    New_Reachable_States = [[] for x in range(T_l.shape[0]*len(A))]
    New_Is_Bridge_State = np.zeros(T_l.shape[0]*len(A))
    New_Bridge_Transitions = [[] for x in range(T_l.shape[0]*len(A))]
    Init = np.zeros((T_l.shape[0]))
    Init = Init.astype(int) 
    

    
    for x in range(len(Acc)):
        for i in range(len(Acc[x][0])):
            for j in range(T_l.shape[0]):
                Is_N_A[len(A)*j + Acc[x][0][i]] = 1
                Which_N_A[len(A)*j + Acc[x][0][i]].append(x)
        
        for i in range(len(Acc[x][1])):
            for j in range(T_l.shape[0]):
                Is_A[len(A)*j + Acc[x][1][i]] = 1
                Which_A[len(A)*j + Acc[x][1][i]].append(x)            

   
    for i in range(T_l.shape[0]):
        for j in range(len(A)):            
            for k in range(T_l.shape[0]):
                for l in range(len(A)):
                    
                    
                    
                    if L[k] in A[j][l]:
                        
                        if j == 0:
                            Init[k] = l
                                                    
                        IA_l[len(A)*i+j, len(A)*k+l] = T_l[i,k]
                        IA_u[len(A)*i+j, len(A)*k+l] = T_u[i,k]
                        

                        
                        if T_u[i,k] > 0:
                            New_Reachable_States[len(A)*i+j].append(len(A)*k+l)
                            if T_l[i,k] == 0:
                                New_Is_Bridge_State[len(A)*i+j] = 1
                                New_Bridge_Transitions[len(A)*i+j].append(len(A)*k+l)
                        
                    else:
                        IA_l[len(A)*i+j, len(A)*k+l] = 0.0
                        IA_u[len(A)*i+j, len(A)*k+l] = 0.0
    
                 

    Is_A = Is_A.astype(int)
    Is_N_A = Is_N_A.astype(int)
    New_Is_Bridge_State = New_Is_Bridge_State.astype(int)                         

    return (IA_l, IA_u, Is_A, Is_N_A, Which_A, Which_N_A, New_Reachable_States, New_Is_Bridge_State, New_Bridge_Transitions, Init) 





def Find_Largest_BSCCs_One_Pair(I_l, I_u, Acc, N_State_Auto, Is_A_State, Is_N_A_State, Which_A_Pair, Which_N_A_Pair, Reachable_States, Is_Bridge_State, Bridge_Transition, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Is_in_P, List_I_UA, List_I_UN, Previous_A_BSCC, Previous_Non_A_BSCC, First_Verif):
    
    

   
           
    G = np.zeros((I_l.shape[0],I_l.shape[1]))
       
    for i in range(I_l.shape[0]):
        for j in range(I_l.shape[1]):
            if I_u[i,j] > 0:
                G[i,j] = 1

    G_prime = np.copy(G)
    

    
    if First_Verif == 0:
        Deleted_States = []
        Prev_A = set().union(*Previous_A_BSCC)
        Prev_N = set().union(*Previous_Non_A_BSCC)
        Deleted_States.extend(list(set(range(G.shape[0])) - set(Prev_A) - set(Prev_N)))
        
        Ind = list(set(Prev_A)|set(Prev_N))
        Ind.sort()
        G = np.delete(np.array(G),Deleted_States,axis=0)
        G = np.delete(np.array(G),Deleted_States,axis=1)
    else:
        Ind = range(G.shape[0])
        
    First_Verif = 0   
    

       

    
    
         
    C,n = SSCC(G)
    
       
    SCC_Status = [0]*n 

    tag = 0
    m = 0

    List_UN = []
    List_UA = []

    Is_In_L_A = np.zeros(I_l.shape[0]) #Is the state in the largest potential accepting BSCC?
    Is_In_L_N_A = np.zeros(I_l.shape[0]) #Is the state in the largest potential non-accepting BSCC?
    Which_A = np.zeros(I_l.shape[0]) #Keeps track of which accepting BSCC does each state belong to (if applicable)
    Which_N_A = np.zeros(I_l.shape[0])
    Which_A.astype(int)
    Which_N_A.astype(int)
    Is_In_L_A.astype(int)
    Is_In_L_N_A.astype(int)
    Potential_Permanent_Accepting = [] #Stores the potential permanent BSCCs until we can check whether it contains a potential component of the other acceptance status
    Potential_Permanent_Accepting_Bridge_States = [] #Stores the potential permanent BSCCs until we can check whether it contains a potential component of the other acceptance status
    Potential_Permanent_Non_Accepting = [] #Stores the potential permanent BSCCs until we can check whether it contains a potential component of the other acceptance status
    Potential_Permanent_Non_Accepting_Bridge_States = [] #Stores the potential permanent BSCCs until we can check whether it contains a potential component of the other acceptance status
        
    Bridge_A = []
    Bridge_N_A = []
        
    while tag == 0:
        
        if len(C) == 0:
            break
        
        SCC = C[m]

              
        
        Orig_SCC = []
        for k in range(len(SCC)):
            Orig_SCC.append(Ind[SCC[k]])

        BSCC = 1

     
        Leak = []
        Check_Tag = 1
        Reach_in_R = [[] for x in range(len(Orig_SCC))]
        Pre = [[] for x in range(len(Orig_SCC))]
        All_Leaks = []
        Check_Orig_SCC = np.zeros(len(Orig_SCC), dtype=int)
        
        
        while (len(Leak) != 0 or Check_Tag == 1):
            
            
            ind_leak = []
            Leak = []
            
            for i in range(len(Orig_SCC)):
                
                if Check_Orig_SCC[i] == -1 : continue
                
                Set_All_Leaks = set(Orig_SCC) - set(All_Leaks)
                Diff_List1 = list(set(Reachable_States[Orig_SCC[i]]) - Set_All_Leaks)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[Orig_SCC[i]]))
                
                if Check_Tag == 1:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[Orig_SCC[i]]) - set(Diff_List1)))
                    for j in range(len(Reach_in_R[i])):
                        Pre[Orig_SCC.index(Reach_in_R[i][j])].append(Orig_SCC[i])
               
                if (len(Diff_List2) != 0) or (sum(I_u[Orig_SCC[i], Reach_in_R[i]])<1) :
                    Leak.append(Orig_SCC[i])
                    ind_leak.append(i)
   
            
            if len(Leak) != 0:
                All_Leaks.extend(Leak)
                BSCC = 0
                for i in range(len(Leak)):
                    Check_Orig_SCC[ind_leak[i]] = -1
                    for j in range(len(Pre[ind_leak[i]])):
                        Reach_in_R[Orig_SCC.index(Pre[ind_leak[i]][j])].remove(Leak[i])
                    
            Check_Tag = 0
           
        if BSCC != 1:    

            SCC = list(set(Orig_SCC) - set(All_Leaks))
            
            for k in range(len(SCC)):
                SCC[k] = Ind.index(SCC[k])
            
            if len(SCC) != 0:
                SCC = sorted(SCC, key=int)                
                New_G = G[np.ix_(SCC,SCC)]
                
                C_new, n_new = SSCC(New_G)
                for j in range(len(C_new)):
                    for k in range(len(C_new[j])):
                        C_new[j][k] = SCC[C_new[j][k]] 
                    C.append(C_new[j])
                    SCC_Status.append(SCC_Status[m])
                    
            
            
        else:  
            
            Bridge_States = []
                
            if SCC_Status[m] == 0:
                
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                
                


                for j in range(len(SCC)):
                    
                    
                    if Is_A_State[Ind[SCC[j]]] == 1:
                        acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_A_Pair[Ind[SCC[j]]][n])
                        ind_acc.append(indices) 

                    if Is_N_A_State[Ind[SCC[j]]] == 1:
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_N_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_N_A_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                        
                      
                    if Is_Bridge_State[Ind[SCC[j]]] == 1:
                                                                                                             
                        Diff_List = np.setdiff1d(Reachable_States[Ind[SCC[j]]], Orig_SCC)                            
                        if len(Diff_List) != 0:
                            Inevitable = 0 
                        Bridge_States.append(Ind[SCC[j]])

                        

                

                
                
                if Inevitable == 1:
                    
                            
                    if len(acc_states) == 0:                        
                        List_I_UN.append([])
                        Leaky_States_P_Non_Accepting.append([])
                        for j in range(len(SCC)):
                            List_I_UN[-1].append(Ind[SCC[j]])                            
                            
                    else:
                        
                        Accept = []
                        Acc_Tag = 0
                        Non_Accept_Remove = [[] for x in range(len(Acc))] #Contains all non-accepting states which prevent the bscc to be accepting for all pairs
                        
                        if len(non_acc_states) == 0:
                            Acc_Tag = 1
                            for j in range(len(acc_states)):
                                Accept.append(acc_states[j])
                        else:        
                            for j in range(len(ind_acc)):
                                for l in range(len(ind_acc[j])):
                                    Check_Tag = 0
                                    Keep_Going = 0
                                    for w in range(len(ind_non_acc)):  
                                        if ind_acc[j][l] in ind_non_acc[w]:
                                            Check_Tag = 1
                                            if len(Non_Accept_Remove[ind_acc[j][l]]) == 0:
                                                Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w])
                                                Keep_Going = 1                                
                                            elif Keep_Going == 0:
                                                break                                
                                    if Check_Tag == 0:                                
                                        Accept.append(acc_states[j])
                                        Acc_Tag = 1  
                        
                        if Acc_Tag == 1:
                            
                            Potential_Permanent_Accepting.append([])
                            Potential_Permanent_Accepting_Bridge_States.append([])                                        
                            for n in range(len(SCC)):
                                Potential_Permanent_Accepting[-1].append(Ind[SCC[n]])
                            for n in range(len(Bridge_States)):
                                Potential_Permanent_Accepting_Bridge_States[-1].append(Bridge_States[n])                                                                                    
                              
                        
                            SCC_bis = [x for x in SCC if x not in Accept]
                            if len(SCC_bis) != 0:
                                SCC_bis = sorted(SCC_bis, key=int)           
                                New_G = G[np.ix(SCC_bis, SCC_bis)]
            
                                C_new, n_new = SSCC(New_G)
                                
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC[C_new[j][k]]
                                    C.append(C_new[j])
                                    SCC_Status.append(2)
                                        
                                                                                
                                                                                                              
                        if Acc_Tag == 0: 
                            
                            Potential_Permanent_Non_Accepting.append([])
                            Potential_Permanent_Non_Accepting_Bridge_States.append([])                                        
                            for n in range(len(SCC)):
                                Potential_Permanent_Non_Accepting[-1].append(Ind[SCC[n]])
                            for n in range(len(Bridge_States)):
                                Potential_Permanent_Non_Accepting_Bridge_States[-1].append(Bridge_States[n])                                                                                    
                            
                            for l in range(len(Non_Accept_Remove)):
                               if len(Non_Accept_Remove[l]) != 0:
                                    SCC_bis = [x for x in SCC if x not in Non_Accept_Remove[l]]
                                    if len(SCC_bis) != 0:
                                        SCC_bis = sorted(SCC_bis, key=int)           
                                        New_G = G[np.ix(SCC_bis, SCC_bis)]
                                        C_new, n_new = SSCC(New_G)
                                        for j in range(len(C_new)):
                                            for k in range(len(C_new[j])):
                                                C_new[j][k] = SCC_bis[C_new[j][k]]   
                                            C.append(C_new[j])
                                            SCC_Status.append(1)                            

                elif len(acc_states) == 0:
                    List_UN.append([])
                    Bridge_N_A.append([])
                    Leaky_States_L_Non_Accepting.append([])
                    for j in range(len(SCC)):
                        List_UN[-1].append(Ind[SCC[j]])
                        Which_N_A[Ind[SCC[j]]] = len(List_UN) - 1
                        Is_In_L_N_A[Ind[SCC[j]]] = 1
                    for x in range(len(Bridge_States)):
                        Bridge_N_A[-1].append(Bridge_States[x])
                            
          

                
                else:
                   
                    Acc_Tag = 0
                    Accept = [] #Contains unmatched accepting states
                    
                                        
                    if len(non_acc_states) == 0:
                        Acc_Tag = 1
                        for j in range(len(acc_states)):
                            Accept.append(acc_states[j])
                    
                    else:
                        Non_Accept_Remove = [[] for x in range(len(Acc))] #Contains all non-accepting states which prevent the bscc to be accepting for all pairs
                        
                        for j in range(len(ind_acc)):
                            for l in range(len(ind_acc[j])):
                                Check_Tag = 0
                                Keep_Going = 0
                                for w in range(len(ind_non_acc)):  
                                    if ind_acc[j][l] in ind_non_acc[w]:
                                        Check_Tag = 1
                                        if len(Non_Accept_Remove[ind_acc[j][l]]) == 0:
                                            Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w])
                                            Keep_Going = 1                                
                                        elif Keep_Going == 0:
                                            break                                
                                if Check_Tag == 0:                                
                                    Accept.append(acc_states[j])
                                    Acc_Tag = 1  
                    
                    
                    if Acc_Tag == 1:
                        List_UA.append([])
                        Bridge_A.append([])
                        Leaky_States_L_Accepting.append([])
                        for n in range(len(SCC)):
                            List_UA[-1].append(Ind[SCC[n]])
                            Which_A[Ind[SCC[n]]] = len(List_UA) - 1
                            Is_In_L_A[Ind[SCC[n]]] = 1
                        
                        for x in range(len(Bridge_States)):
                            Bridge_A[-1].append(Bridge_States[x])
                        

                        SCC_bis = [x for x in SCC if x not in Accept]
                        if len(SCC_bis) != 0:
                            SCC_bis = sorted(SCC_bis, key=int)           

                            New_G = G[np.ix(SCC_bis,SCC_bis)]
        
                            C_new, n_new = SSCC(New_G)
                            
                            for j in range(len(C_new)):
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC[C_new[j][k]]
                                C.append(C_new[j])
                                SCC_Status.append(2)
                                
                    else:
                        
                        List_UN.append([])
                        Bridge_N_A.append([])
                        Leaky_States_L_Non_Accepting.append([])
                        for n in range(len(SCC)):
                            List_UN[-1].append(Ind[SCC[n]])
                            Which_N_A[Ind[SCC[n]]] = len(List_UN) - 1
                            Is_In_L_N_A[Ind[SCC[n]]] = 1
                       
                        for x in range(len(Bridge_States)):
                            Bridge_N_A[-1].append(Bridge_States[x])
                        
                        for l in range(len(Non_Accept_Remove)):
                           if len(Non_Accept_Remove[l]) != 0:
                                SCC_bis = [x for x in SCC if x not in Non_Accept_Remove[l]]
                                if len(SCC_bis) != 0:
                                    SCC_bis = sorted(SCC_bis, key=int)           

                                    New_G = G[np.ix_(SCC_bis, SCC_bis)]
                                    C_new, n_new = SSCC(New_G)
                                    for j in range(len(C_new)):
                                        for k in range(len(C_new[j])):
                                            C_new[j][k] = SCC_bis[C_new[j][k]]   
                                        C.append(C_new[j])
                                        SCC_Status.append(1)
                        
  
              
            if SCC_Status[m] == 1: ###This BSCC is part of a Potentially Larger Non-A BSCC, Want to check if it contains a potential accepting BSCC 
                 
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                                        
                for j in range(len(SCC)):
                    
                    if Is_A_State[Ind[SCC[j]]] == 1:
                        acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_A_Pair[Ind[SCC[j]]][n])
                        ind_acc.append(indices) 

                    if Is_N_A_State[Ind[SCC[j]]] == 1:
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_N_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_N_A_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices) 
                             
                    if Is_Bridge_State[Ind[SCC[j]]] == 1:
                        Bridge_States.append(Ind[SCC[j]])
                
                
                if len(acc_states) > 0:
                    Acc_Tag = 0
                    
                    if len(non_acc_states) == 0:
                        Acc_Tag = 1
                    else:     
                        Non_Accept_Remove = [[] for x in range(len(Acc))] #Contains all non-accepting states which prevent the bscc to be accepting for all pairs
                        
                        for j in range(len(ind_acc)):
                            for l in range(len(ind_acc[j])):
                                Check_Tag = 0
                                Keep_Going = 0
                                for w in range(len(ind_non_acc)):  
                                    if ind_acc[j][l] in ind_non_acc[w]:
                                        Check_Tag = 1
                                        if len(Non_Accept_Remove[ind_acc[j][l]]) == 0:
                                            Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w])
                                            Keep_Going = 1                                
                                        elif Keep_Going == 0:
                                            break                                
                                if Check_Tag == 0:                                
                                    Acc_Tag = 1  
                                                                    
                    if Acc_Tag == 1:
                        
                        
                        List_UA.append([])
                        Bridge_A.append([])
                        Leaky_States_L_Accepting.append([])
                        for j in range(len(SCC)):
                            List_UA[-1].append(Ind[SCC[j]])
                            Which_A[Ind[SCC[j]]] = len(List_UA) - 1
                            Is_In_L_A[Ind[SCC[j]]] = 1
                        for x in range(len(Bridge_States)):
                            Bridge_A[-1].append(Bridge_States[x])  
                            
                    else:    
                        
                        for l in range(len(Non_Accept_Remove)):
                           if len(Non_Accept_Remove[l]) != 0:
                                SCC_bis = [x for x in SCC if x not in Non_Accept_Remove[l]]
                                if len(SCC_bis) != 0:
                                    SCC_bis = sorted(SCC_bis, key=int)           

                                    New_G = G[np.ix_(SCC_bis, SCC_bis)]
                                    C_new, n_new = SSCC(New_G)
                                    for j in range(len(C_new)):
                                        for k in range(len(C_new[j])):
                                            C_new[j][k] = SCC_bis[C_new[j][k]]   
                                        C.append(C_new[j])
                                        SCC_Status.append(1)
             
            if SCC_Status[m] == 2: ###This SCC is part of a Potentially Larger A BSCC, Want to check if it contains a potential non-accepting BSCC
                 
                ind_acc = []
                ind_non_acc = []
                acc_states = []
                non_acc_states = []
                
                
                for j in range(len(SCC)):
                    
                    
                    if Is_A_State[Ind[SCC[j]]] == 1:
                        acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_A_Pair[Ind[SCC[j]]][n])
                        ind_acc.append(indices) 

                    if Is_N_A_State[Ind[SCC[j]]] == 1:
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_N_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_N_A_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         

                    if Is_Bridge_State[Ind[SCC[j]]] == 1:                                         
                            Bridge_States.append(Ind[SCC[j]])

                                      
                
                if len(acc_states) == 0:
                    
                    
                    List_UN.append([])
                    Bridge_N_A.append([])
                    Leaky_States_L_Non_Accepting.append([])
                    for j in range(len(SCC)):
                        List_UN[-1].append(Ind[SCC[j]])
                        Which_N_A[Ind[SCC[j]]] = len(List_UN) - 1
                        Is_In_L_N_A[Ind[SCC[j]]] = 1
                    for x in range(len(Bridge_States)):
                        Bridge_N_A[-1].append(Bridge_States[x])
                    
                
                else:   
                    
                        Acc_Tag = 0
                        Accept = [] #Contains unmatched accepting states
                        for j in range(len(ind_acc)):                            
                            for l in range(len(ind_acc[j])):
                                if Accept[-1] == acc_states[j]:
                                    break
                                Check_Tag = 0
                                for w in range(len(ind_non_acc)):  
                                    if ind_acc[j][l] in ind_non_acc[w]:
                                        Check_Tag = 1
                                        break                                
                                if Check_Tag == 0:                                
                                    Accept.append(acc_states[j])
                                    Acc_Tag = 1                         
    
    
                        if Acc_Tag == 0:
                            
                            
                            List_UN.append([])
                            Bridge_N_A.append([])
                            Leaky_States_L_Non_Accepting.append([])
                            for j in range(len(SCC)):
                                List_UN[-1].append(Ind[SCC[j]])
                                Which_N_A[Ind[SCC[j]]] = len(List_UN) - 1
                                Is_In_L_N_A[Ind[SCC[j]]] = 1
                            for x in range(len(Bridge_States)):
                                Bridge_N_A[-1].append(Bridge_States[x])
                            
                        else:
                             
                            SCC_bis = [x for x in SCC if x not in Accept]
                            if len(SCC_bis) != 0:
                                SCC_bis = sorted(SCC_bis, key=int)           
                                New_G = G[np.ix_(SCC_bis, SCC_bis)]
            
                                C_new, n_new = SSCC(New_G)
                                
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC[C_new[j][k]]
                                    C.append(C_new[j])
                                    SCC_Status.append(2)
                
                
        m +=1
        if m == len(C): tag = 1
    
    
    for i in range(len(Potential_Permanent_Accepting)):
        Check = 0
        for j in range(len(Potential_Permanent_Accepting[i])):
            if Is_In_L_N_A[Potential_Permanent_Accepting[i][j]] == 1:
                Check = 1
                break
        
        if Check == 0: 
            List_I_UA.append(Potential_Permanent_Accepting[i])
            Leaky_States_P_Accepting.append([])                       

        
        else: 
            List_UA.append([])
            Bridge_A.append(Potential_Permanent_Accepting_Bridge_States[i])
            Leaky_States_L_Accepting.append([])
            for n in range(len(Potential_Permanent_Accepting[i])):
                List_UA[-1].append(Potential_Permanent_Accepting[i][n])
                Which_A[Potential_Permanent_Accepting[i][n]] = len(List_UA) - 1
                Is_In_L_A[Potential_Permanent_Accepting[i][n]] = 1
    
            
    for i in range(len(Potential_Permanent_Non_Accepting)):
        Check = 0       
        for j in range(len(Potential_Permanent_Non_Accepting[i])):
            if Is_In_L_A[Potential_Permanent_Non_Accepting[i][j]] == 1:
                Check = 1
                break        
        if Check == 0:
            List_I_UN.append(Potential_Permanent_Non_Accepting[i])
            Leaky_States_P_Non_Accepting.append([])                     
      
        else:            
            List_UN.append([])
            Bridge_N_A.append(Potential_Permanent_Non_Accepting_Bridge_States[i])
            Leaky_States_L_Non_Accepting.append([])
            for n in range(len(Potential_Permanent_Non_Accepting[i])):
                List_UN[-1].append(Potential_Permanent_Non_Accepting[i][n])
                Which_N_A[Potential_Permanent_Non_Accepting[i][n]] = len(List_UN) - 1
                Is_In_L_N_A[Potential_Permanent_Non_Accepting[i][n]] = 1

    Which_A = Which_A.astype(int)
    Which_N_A = Which_N_A.astype(int)
    Is_In_L_A = Is_In_L_A.astype(int)
    Is_In_L_N_A = Is_In_L_N_A.astype(int)
    
    return (List_UN, List_UA, List_I_UN, List_I_UA, Is_In_L_A, Is_In_L_N_A, Which_A, Which_N_A, Bridge_A, Bridge_N_A, G_prime, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting)





def Find_Winning_Losing_Components(I_l, I_u, List_L_N_A, List_L_A, List_I_N_A, List_I_A, Reachable_States, Is_Bridge_State, Bridge_Transition, G, Bridge_Acc, Bridge_N_Acc, Is_State_In_L_A, Is_State_In_L_N_A, Is_in_P, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC):

    
    WC_L = [[] for i in range(len(List_L_A))]
    WC_I = []
    Is_in_WC_L = np.zeros(G.shape[0])
    Which_WC_L = [[] for i in range(G.shape[0])] #First number in list tells you which BSCC, second number tells you which component around the BSCC
    Bridge_WC_L = [[] for i in range(len(List_L_A))]
    Is_in_LC_L = np.zeros(G.shape[0])
    Which_LC_L = [[] for i in range(G.shape[0])] #First number in list tells you which BSCC, second number tells you which component around the BSCC
    Bridge_LC_L = [[] for i in range(len(List_L_N_A))]
    
    
    Is_in_WC_P = np.zeros(G.shape[0]) #Is the state in a potential winning component around a permanent winning component?
    Which_WC_P = [[] for i in range(G.shape[0])]
    Bridge_WC_P = [[] for i in range(len(List_I_A))]
    Is_in_LC_P = np.zeros(G.shape[0]) #Is the state in a potential losing component around a permanent losing component?
    Which_LC_P = [[] for i in range(G.shape[0])]
    Bridge_LC_P = [[] for i in range(len(List_I_N_A))]
    LC_L = [[] for i in range(len(List_L_N_A))]
    LC_I = []
    
    
    G_original = np.copy(G)
    

    
    for n in range(len(List_I_A)):
             
        
        T = List_I_A[n]
        G = np.copy(G_original)
   
        C = []
        W = []
        m = 0
        
        if len(Leaky_States_P_Accepting[n]) == 0:
            for q in range(len(Previous_A_BSCC)):
                if T[0] in Previous_A_BSCC[q]:
                    Leaky_States_P_Accepting[n] = list(Leaky_L_Accepting_Previous[q])  
                    break        

        Ind = [x for x in (range(G.shape[0])) if x not in Leaky_States_P_Accepting[n]]      

        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i
        
        G = np.delete(np.asarray(G), Leaky_States_P_Accepting[n], axis = 0)
        G = np.delete(np.asarray(G), Leaky_States_P_Accepting[n], axis = 1)
        
        while len(C) != 0 or m == 0:                        
            Gr = igraph.Graph.Adjacency(G.tolist())
            
            if m != 0:
                R_prev = list(R)             
            
            R = []
            
            for q in range(len(T)):                 
                Res = Gr.subcomponent(Indices[T[q]], mode="IN")
                R2 = [x for x in Res if x not in R]
                R.extend(R2)
 
            for q in range(len(R)):
                R[q] = Ind[R[q]] #Converting back to original indices 
            
            if m == 0:             
                Tr = set(range(G_original.shape[0])) - set(R)                   
            else:
                Tr = Tr | (set(R_prev) - set(R))
                
            R2 = list(R)
            R = list( set(R) - set(T) )
            
            C = []
            Is_In_C = np.zeros(G_original.shape[0])
            Leaks = list(R2)
            
            if m == 0:
           
                Ind_Original_R = np.zeros(G_original.shape[0], dtype=int)
                for i in range(len(R)):
                    Ind_Original_R[R[i]] = i
                Reach_in_R = [[] for x in range(len(R))]
                Pre = [[] for x in range(len(R))]
                W = []
            
            
            ind_leak = []
            C = []

            
            for i in range(len(R)):
                                
                Diff_List1 = set(Reachable_States[R[i]]) - set(R2)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[R[i]]))
                
                if m == 0:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[R[i]]) - Diff_List1 - set(T) ))
                    for j in range(len(Reach_in_R[i])):
                        Pre[R.index(Reach_in_R[i][j])].append(R[i])
                    Reach_in_R[i].extend(set.intersection(set(Reachable_States[R[i]]), set(T)))   
               
                if (len(Diff_List2) != 0) or (sum(I_u[R[i], Reach_in_R[Ind_Original_R[R[i]]]])<1) :
                    C.append(R[i])
                    ind_leak.append(i)
   
            
            if len(C) != 0:
                W.extend(C)
                for i in range(len(C)):
                    for j in range(len( Pre[Ind_Original_R[C[i]]] )):
                        Reach_in_R[Ind_Original_R[Pre[Ind_Original_R[C[i]]][j]]].remove(C[i])
                    
                    
                
            C_tran  = []
            for i in range(len(C)):
                C_tran.append(Ind.index(C[i]))
            
            for i in range(len(C_tran)):
                G[C_tran[i],:] = np.zeros(G.shape[1])
                G[C_tran[i],C_tran[i]] = 1
            
            List = list(set(range(G.shape[0])) - set(C_tran))
            
            for i in range(len(List)):
                for j in range(len(C_tran)):
                    G[List[i],C_tran[j]] = 0
            
                                                                    
            m += 1
        
        Leaky_States_P_Accepting[n] = list(set(Tr) | set(W))       
        R = list(set(range(G_original.shape[0])) - set(W))
        
        R = list(set(R) - set(Tr))
        B = list(R)
        
        for q in range(len(R)):
            Is_in_WC_L[R[q]] = 1
                    
               
            
        Remove = []    
        for i in range(len(R)): 
            if Is_State_In_L_N_A[R[i]] == 1:
                Remove.append(R[i])      
        R = list(set(R) - set(Remove))


        
        C = []
        m = 0
        Is_In_C = np.zeros(G_original.shape[0])

        for i in range(len(R)):
            
            if R[i] in (C+T): continue           
            leak = 0
            Leaky_State = []
            Diff_List = np.setdiff1d(Reachable_States[R[i]], R)
            
            

            if (len(Diff_List) != 0):
                leak = 1
                C.append(R[i])
                Is_In_C[R[i]] = 1
                Leaky_State.append(R[i])
                #break
                                                                                     
            if leak == 1:
                
                tag2 = 0
                k = 0
                while tag2 == 0:
                                             
                    for j in range(len(R)):
                        if Is_In_C[R[j]] == 0:
                            if Leaky_State[k] in Reachable_States[R[j]]:
                                C.append(R[j])
                                Is_In_C[R[j]] = 1
                                Leaky_State.append(R[j])
                                               
                    k += 1                   
                    if k == len(Leaky_State): tag2 = 1
                   

                
        List_Permanents = np.setdiff1d(R, C).tolist()
        

        for i in range(len(List_Permanents)):
            Is_in_P[List_Permanents[i]] = 1
        
        List_I_A[n] = list(List_Permanents)
        
        
        WC_I = WC_I + List_Permanents
            
        B = list(set(B) - set(List_Permanents))                                                       
        
        G = np.copy(G_original)
        G = G[np.ix_(B, B)]
        
        num, D = scipy.sparse.csgraph.connected_components(G, directed = False)
        W = [[] for i in range(num) ]
        for q in range(len(D)):
            for i in range(num):
                if D[q] == i:
                    W[i].append(B[q])
                    break
        
        for q in range(len(W)):
            Set_C = W[q]            
            Bridges = []
            for l in range(len(Set_C)):
                Is_in_WC_P[Set_C[l]] = 1
                Which_WC_P[Set_C[l]].append([n,q])
                if Is_Bridge_State[Set_C[l]] == 1:
                    Bridges.append(Set_C[l])
            Bridge_WC_P[n].append(Bridges) 
            
            
            
    
    for n in range(len(List_L_A)):
        
        
        T = List_L_A[n]
             
        Already_Check = 1 #Checks if the BSCC already belongs to a larger potential component. If that is the case, the BSCC doesn't need to be rechecked
        for q in range(len(T)):
            if Is_in_WC_L[T[q]] == 1:
                Already_Check = 0
                break
                   
        if Already_Check == 0:
            continue
        
        for q in range(len(Previous_A_BSCC)):
            if T[0] in Previous_A_BSCC[q]:
                Leaky_States_L_Accepting[n] = list(Leaky_L_Accepting_Previous[q])
                break
         
                        
        G = np.copy(G_original)
        C = []
        W = []
        m = 0
        
        Ind = [x for x in (range(G.shape[0])) if x not in Leaky_States_L_Accepting[n]]            
        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i 
            
            
        G = np.delete(np.asarray(G), Leaky_States_L_Accepting[n], axis = 0)
        G = np.delete(np.asarray(G), Leaky_States_L_Accepting[n], axis = 1)
                
        while len(C) != 0 or m == 0:                        
            Gr = igraph.Graph.Adjacency(G.tolist())
            
            if m != 0:
                R_prev = list(R)
            
            R = []
            
            for q in range(len(T)):                 
                Res = Gr.subcomponent(Indices[T[q]], mode="IN")
                R2 = [x for x in Res if x not in R]
                R.extend(R2)
 
            for q in range(len(R)):
                R[q] = Ind[R[q]] #Converting back to original indices 
            
            if m == 0:             
                Tr = set(range(G_original.shape[0])) - set(R)                   
            else:
                Tr = Tr | (set(R_prev) - set(R))
                
            R2 = list(R)
            R = list( set(R) - set(T) )
            
            C = []
            Is_In_C = np.zeros(G_original.shape[0])
            Leaks = list(R2)
            
            if m == 0:
           
                Ind_Original_R = np.zeros(G_original.shape[0], dtype=int)
                for i in range(len(R)):
                    Ind_Original_R[R[i]] = i
                Reach_in_R = [[] for x in range(len(R))]
                Pre = [[] for x in range(len(R))]
                W = []
            
            
            ind_leak = []
            C = []

            
            for i in range(len(R)):
                                
                Diff_List1 = set(Reachable_States[R[i]]) - set(R2)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[R[i]]))
                
                if m == 0:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[R[i]]) - Diff_List1 - set(T) ))
                    for j in range(len(Reach_in_R[i])):
                        Pre[R.index(Reach_in_R[i][j])].append(R[i])
                    Reach_in_R[i].extend(set.intersection(set(Reachable_States[R[i]]), set(T)))   
               
                if (len(Diff_List2) != 0) or (sum(I_u[R[i], Reach_in_R[Ind_Original_R[R[i]]]])<1) :
                    C.append(R[i])
                    ind_leak.append(i)
   
            
            if len(C) != 0:
                W.extend(C)
                for i in range(len(C)):
                    for j in range(len( Pre[Ind_Original_R[C[i]]] )):
                        Reach_in_R[Ind_Original_R[Pre[Ind_Original_R[C[i]]][j]]].remove(C[i])
                
                
            C_tran  = []
            for i in range(len(C)):
                C_tran.append(Ind.index(C[i]))
            
            for i in range(len(C_tran)):
                G[C_tran[i],:] = np.zeros(G.shape[1])
                G[C_tran[i],C_tran[i]] = 1
            
            List = list(set(range(G.shape[0])) - set(C_tran))
            
            for i in range(len(List)):
                for j in range(len(C_tran)):
                    G[List[i],C_tran[j]] = 0
            
            m += 1        
        
        
        Leaky_States_L_Accepting[n] = list(Tr | set(W))
        L = set(range(G_original.shape[0])) - set(W)
        List_Comp = list(L - set(Tr))
        


        WC_L[n].append(List_Comp)
        
        for q in range(len(WC_L[n][-1])):
            Is_in_WC_L[WC_L[n][-1][q]] = 1
        
        WC_L[n].pop()
        
        for q in range(len(T)):
            Which_WC_L[T[q]].append([n,0])
        WC_L[n].append(T)
        Bridge_WC_L[n].append(Bridge_Acc[n])
                
        Win_Comp = list(set(List_Comp) - set(T)) 
        
        
        
        G = np.copy(G_original)
        G = G[np.ix_(Win_Comp, Win_Comp)] 

        num, D = scipy.sparse.csgraph.connected_components(G, directed = False)
        W = [[] for i in range(num)]

        for q in range(len(D)):
            for i in range(num):
                if D[q] == i:
                    W[i].append(Win_Comp[q])
                    break


        for q in range(len(W)):
            Set_C = W[q]           
            WC_L[n].append(Set_C)
            Bridges = list(Bridge_Acc[n])
            for l in range(len(Set_C)):
                Which_WC_L[Set_C[l]].append([n,q+1])
                if Is_Bridge_State[Set_C[l]] == 1:
                    Bridges.append(Set_C[l])
            Bridge_WC_L[n].append(Bridges)
            
        
     
        
    for n in range(len(List_I_N_A)):
        
        T = List_I_N_A[n]
        G = np.copy(G_original)

        if len(Leaky_States_P_Non_Accepting[n]) == 0:
            for q in range(len(Previous_Non_A_BSCC)):
                if T[0] in Previous_Non_A_BSCC[q]:
                    Leaky_States_P_Non_Accepting[n] = list(Leaky_L_Non_Accepting_Previous[q])
                    break

        Ind = [x for x in (range(G.shape[0])) if x not in Leaky_States_P_Non_Accepting[n]]            
        
        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i         
        
        G = np.delete(np.asarray(G), Leaky_States_P_Non_Accepting[n], axis = 0)
        G = np.delete(np.asarray(G), Leaky_States_P_Non_Accepting[n], axis = 1)
        
        C = []
        W = []
        m = 0
                
        while len(C) != 0 or m == 0:                        
            Gr = igraph.Graph.Adjacency(G.tolist())
            
            if m != 0:
                R_prev = list(R)             
            
            R = []
            
            for q in range(len(T)):                 
                Res = Gr.subcomponent(Indices[T[q]], mode="IN")
                R2 = [x for x in Res if x not in R]
                R.extend(R2)
 
            for q in range(len(R)):
                R[q] = Ind[R[q]] #Converting back to original indices 
            
            if m == 0:             
                Tr = set(range(G_original.shape[0])) - set(R)                   
            else:
                Tr = Tr | (set(R_prev) - set(R))
                
            R2 = list(R)
            R = list( set(R) - set(T) ) #Removing the states T from the states which can reach T
            
            C = []
            Is_In_C = np.zeros(G_original.shape[0])
            Leaks = list(R2)
            
            if m == 0:
           
                Ind_Original_R = np.zeros(G_original.shape[0], dtype=int)
                for i in range(len(R)):
                    Ind_Original_R[R[i]] = i
                Reach_in_R = [[] for x in range(len(R))]
                Pre = [[] for x in range(len(R))]
                W = []
            
            
            ind_leak = []
            C = []

            
            for i in range(len(R)):
                                
                Diff_List1 = set(Reachable_States[R[i]]) - set(R2)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[R[i]]))
                
                if m == 0:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[R[i]]) - Diff_List1 - set(T) ))
                    for j in range(len(Reach_in_R[i])):
                        Pre[R.index(Reach_in_R[i][j])].append(R[i])
                    Reach_in_R[i].extend(set.intersection(set(Reachable_States[R[i]]), set(T)))   
               
                if (len(Diff_List2) != 0) or (sum(I_u[R[i], Reach_in_R[Ind_Original_R[R[i]]]])<1) :
                    C.append(R[i])
                    ind_leak.append(i)
   
            
            if len(C) != 0:
                W.extend(C)
                for i in range(len(C)):
                    for j in range(len( Pre[Ind_Original_R[C[i]]] )):
                        Reach_in_R[Ind_Original_R[Pre[Ind_Original_R[C[i]]][j]]].remove(C[i])
                    
                    
                
            C_tran  = []
            for i in range(len(C)):
                C_tran.append(Ind.index(C[i]))
            
            for i in range(len(C_tran)):
                G[C_tran[i],:] = np.zeros(G.shape[1])
                G[C_tran[i],C_tran[i]] = 1
            
            List = list(set(range(G.shape[0])) - set(C_tran))
            
            for i in range(len(List)):
                for j in range(len(C_tran)):
                    G[List[i],C_tran[j]] = 0
            
                                                                    
            m += 1
        
        Leaky_States_P_Non_Accepting[n] = list(set(Tr) | set(W))       
        R = list(set(range(G_original.shape[0])) - set(W))
        R = list(set(R) - set(Tr))
        B = list(R)    #Save the largest potential components around the BSCC    
        for q in range(len(R)):
            Is_in_LC_L[R[q]] = 1            


        Remove = []    
        for i in range(len(R)): 
            if Is_State_In_L_A[R[i]] == 1:
                Remove.append(R[i])
        
        R = list(set(R) - set(Remove))
            
        C = []
        m = 0
        Is_In_C = np.zeros(G_original.shape[0])
               
        for i in range(len(R)):
            
            if R[i] in (C+T): continue           
            leak = 0
            Leaky_State = []
            Diff_List = np.setdiff1d(Reachable_States[R[i]], R)

            if (len(Diff_List) != 0):
                leak = 1
                C.append(R[i])
                Is_In_C[R[i]] = 1
                Leaky_State.append(R[i])
                #break
                                                                                     
            if leak == 1:
                
                tag2 = 0
                k = 0
                while tag2 == 0:
                                             
                    for j in range(len(R)):
                        if Is_In_C[R[j]] == 0:
                            if Leaky_State[k] in Reachable_States[R[j]]:
                                C.append(R[j])
                                Is_In_C[R[j]] = 1
                                Leaky_State.append(R[j])
                                               
                    k += 1                   
                    if k == len(Leaky_State): tag2 = 1
                   

                
         
        List_Permanents = np.setdiff1d(R, C).tolist() 
        for i in range(len(List_Permanents)):
            Is_in_P[List_Permanents[i]] = 1                  
            
        List_I_N_A[n] = list(List_Permanents)
            
        LC_I = LC_I + List_Permanents 
        
        B = list(set(B) - set(List_Permanents))

        G = np.copy(G_original)
        G = G[np.ix_(B, B)]
        
        num, D = scipy.sparse.csgraph.connected_components(G, directed = False)
        W = [[] for i in range(num) ]
        for q in range(len(D)):
            for i in range(num):
                if D[q] == i:
                    W[i].append(B[q])
                    break
        
        for q in range(len(W)):
            Set_C = W[q]            
            Bridges = []
            for l in range(len(Set_C)):
                Is_in_LC_P[Set_C[l]] = 1
                Which_LC_P[Set_C[l]].append([n,q])
                if Is_Bridge_State[Set_C[l]] == 1:
                    Bridges.append(Set_C[l])
            Bridge_LC_P[n].append(Bridges)             
   
    
    for n in range(len(List_L_N_A)):
        
        T = List_L_N_A[n]
        
        Already_Check = 1 #Checks if the BSCC already belongs to a larger potential component. If that is the case, the BSCC doesn't need to be rechecked
        for q in range(len(T)):
            if Is_in_LC_L[T[q]] == 1:
                Already_Check = 0
                break
       
        if Already_Check == 0:
            continue

        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i 
       
        for q in range(len(Previous_Non_A_BSCC)):
            if T[0] in Previous_Non_A_BSCC[q]:
                Leaky_States_L_Non_Accepting[n] = list(Leaky_L_Non_Accepting_Previous[q])
                break
                  
        G = np.copy(G_original)
        C = []
        W = []
        m = 0
        
        Ind = [x for x in (range(G.shape[0])) if x not in Leaky_States_L_Non_Accepting[n]]            
        
        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i
            
        G = np.delete(np.asarray(G), Leaky_States_L_Non_Accepting[n], axis = 0)
        G = np.delete(np.asarray(G), Leaky_States_L_Non_Accepting[n], axis = 1)
        
        while len(C) != 0 or m == 0:    
                    
            Gr = igraph.Graph.Adjacency(G.tolist())           
            if m != 0:
                R_prev = list(R)
            
            R = []
            
            for q in range(len(T)):                 
                Res = Gr.subcomponent(Indices[T[q]], mode="IN")
                R2 = [x for x in Res if x not in R]
                R.extend(R2)
 
            for q in range(len(R)):
                R[q] = Ind[R[q]] #Converting back to original indices 
            
            if m == 0:             
                Tr = set(range(G_original.shape[0])) - set(R)                   
            else:
                Tr = Tr | (set(R_prev) - set(R))
                
            R2 = list(R)
            R = list( set(R) - set(T) )
            
            C = []
            Is_In_C = np.zeros(G_original.shape[0])
            
            if m == 0:
           
                Ind_Original_R = np.zeros(G_original.shape[0], dtype=int)
                for i in range(len(R)):
                    Ind_Original_R[R[i]] = i
                Reach_in_R = [[] for x in range(len(R))]
                Pre = [[] for x in range(len(R))]
                W = []
            
            
            ind_leak = []
            C = []

            
            for i in range(len(R)):
                                
                Diff_List1 = set(Reachable_States[R[i]]) - set(R2)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[R[i]]))
                
                if m == 0:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[R[i]]) - Diff_List1 - set(T) ))
                    for j in range(len(Reach_in_R[i])):
                        Pre[R.index(Reach_in_R[i][j])].append(R[i])
                    Reach_in_R[i].extend(set.intersection(set(Reachable_States[R[i]]), set(T)))   
               
                if (len(Diff_List2) != 0) or (sum(I_u[R[i], Reach_in_R[Ind_Original_R[R[i]]]])<1) :
                    C.append(R[i])
                    ind_leak.append(i)
   
            
            if len(C) != 0:
                W.extend(C)
                for i in range(len(C)):
                    for j in range(len( Pre[Ind_Original_R[C[i]]] )):
                        Reach_in_R[Ind_Original_R[Pre[Ind_Original_R[C[i]]][j]]].remove(C[i])
                
                
            C_tran  = []
            for i in range(len(C)):
                C_tran.append(Ind.index(C[i]))
            
            for i in range(len(C_tran)):
                G[C_tran[i],:] = np.zeros(G.shape[1])
                G[C_tran[i],C_tran[i]] = 1
            
            List = list(set(range(G.shape[0])) - set(C_tran))
            
            for i in range(len(List)):
                for j in range(len(C_tran)):
                    G[List[i],C_tran[j]] = 0
            
            m += 1        
        
        
        Leaky_States_L_Non_Accepting[n] = list(Tr | set(W))
        L = set(range(G_original.shape[0])) - set(W)
        List_Comp = list(L - set(Tr))       
       
        
        LC_L[n].append(List_Comp)
        
        for q in range(len(LC_L[n][-1])):
            Is_in_LC_L[LC_L[n][-1][q]] = 1
        

        LC_L[n].pop()
        
        for q in range(len(T)):
            Which_LC_L[T[q]].append([n,0])
        LC_L[n].append(T)
        Bridge_LC_L[n].append(Bridge_N_Acc[n])
        
        
        #Below, we want to check whether sets of Losing components are disjoint        
        Los_Comp = list(set(List_Comp) - set(T)) 
        
        # Find sets of disjoint Components W around the BSCC
        
        
        G = np.copy(G_original)
        G = G[np.ix_(Los_Comp, Los_Comp)]        
        num, D = scipy.sparse.csgraph.connected_components(G, directed = False)
        W = [[] for i in range(num) ]

        for q in range(len(D)):
            for i in range(num):
                if D[q] == i:
                    W[i].append(Los_Comp[q])
                    break
        for q in range(len(W)):
            Set_C = W[q]            
            LC_L[n].append(Set_C)
            Bridges = list(Bridge_N_Acc[n])
            for l in range(len(Set_C)):
                Which_LC_L[Set_C[l]].append([n,q+1])
                if Is_Bridge_State[Set_C[l]] == 1:
                    Bridges.append(Set_C[l])
            Bridge_LC_L[n].append(Bridges)
  

        
    All_WC_L = []   
    All_LC_L = []      
        
    for n in range(Is_in_WC_L.shape[0]):
        if Is_in_WC_L[n] == 1:
            All_WC_L.append(n)
    
    for n in range(Is_in_LC_L.shape[0]):
        if Is_in_LC_L[n] == 1:
            All_LC_L.append(n)
            
    for n in range(len(LC_I)):
        Is_in_LC_L[LC_I[n]] = 0
        
    for n in range(len(WC_I)):
        Is_in_WC_L[WC_I[n]] = 0
        
    for n in range(len(Is_in_WC_P)):
        if Is_in_WC_P[n] == 1:
            Is_in_WC_L[n] = 0
    
        if Is_in_LC_P[n] == 1:
            Is_in_LC_L[n] = 0 
            
            
    Previous_A_BSCC = copy.deepcopy(List_L_A)
    Previous_Non_A_BSCC = copy.deepcopy(List_L_N_A)
    Leaky_L_Accepting_Previous = copy.deepcopy(Leaky_States_L_Accepting)
    Leaky_L_Non_Accepting_Previous = copy.deepcopy(Leaky_States_L_Non_Accepting)
    

    return WC_L, WC_I, LC_L, LC_I, All_WC_L, All_LC_L, Is_in_WC_L, Which_WC_L, Bridge_WC_L, Is_in_LC_L, Which_LC_L, Bridge_LC_L, Is_in_WC_P, Which_WC_P, Bridge_WC_P, Is_in_LC_P, Which_LC_P, Bridge_LC_P, Is_in_P, List_I_A, List_I_N_A, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC




def Reachability_Upper(IA_l, IA_u, Q1, Q0, Num_States, Automata_size, Reach, Init):
    
    
    Descending_Order = []
    Index_Vector = np.zeros((IA_l.shape[0],1))  
      
    for k in range(IA_l.shape[0]):
                             
        if k in Q1:       

            Index_Vector[k,0] = 1.0
            Descending_Order.insert(0,k)
            
        elif k not in Q0:
            
            Index_Vector[k,0] = 0.0
            Descending_Order.append(k) 
            
    for k in range(len(Q0)):
        Descending_Order.append(Q0[k])
    
    
    d = {k:v for v,k in enumerate(Descending_Order)} 
    Sort_Reach = []

 
    for i in range(len(Reach)):
        Reach[i].sort(key=d.get)
        Sort_Reach.append(Reach[i])
    
 

    Phi_Max = Phi_Computation_Upper(IA_u, IA_l, Descending_Order, Q1, Q0, Reach, Sort_Reach)
    Steps_High = np.dot(Phi_Max, Index_Vector)
   
    for i in range(len(Q1)):    
        Steps_High[Q1[i]][0] = 1.0
    for i in range(len(Q0)): 
        Steps_High[Q0[i]][0] = 0.0
    

           
    Success_Intervals = []    
    for i in range(IA_l.shape[0]):       
        Success_Intervals.append(Steps_High[i][0])   
    
    Terminate_Check = 0
    Convergence_threshold = 0.000000001
      
    while Terminate_Check == 0:
        
        
        Previous_List = copy.copy(Descending_Order)         
              
        
        for i in range(len(Q0)):
            Success_Intervals[Q0[i]] = 0.0
        
        for i in range(len(Q1)):
            Success_Intervals[Q1[i]] = 1.0
       
        Descending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Descending_Order = list(Descending_Order[(-Success_Array).argsort()])
        
        d = {k:v for v,k in enumerate(Descending_Order)}
        Sort_Reach = []
        for i in range(len(Reach)):
            Reach[i].sort(key=d.get)
            Sort_Reach.append(Reach[i])
            
        
        if Previous_List != Descending_Order:
            Phi_Max = Phi_Computation_Upper(IA_u, IA_l, Descending_Order, Q1, Q0, Reach, Sort_Reach)
        

         
        Steps_High = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Max), Steps_High)

    
        for i in range(len(Q1)):  
            Steps_High[Q1[i]][0] = 1.0

            
        for i in range(len(Q0)):   
            Steps_High[Q0[i]][0] = 0.0 
            
   
                  
        Max_Difference = 0
               
        
        for i in range(IA_l.shape[0]):
           
            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Steps_High[i][0]))            
            Success_Intervals[i] = Steps_High[i][0]
            
        
        if Max_Difference < Convergence_threshold:       
            Terminate_Check = 1
     

            
    Bounds = []
    Prod_Bounds = []

    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
        
    for i in range(len(Success_Intervals)):
        Prod_Bounds.append(Success_Intervals[i]) 
       
    return (Bounds, Prod_Bounds, Phi_Max)





def Phi_Computation_Upper(Upper, Lower, Order_D, q1, q0, Reach, Reach_Sort):

    Phi_max = np.zeros((Upper.shape[0], Upper.shape[1]))   
    
    
    
    for j in range(Upper.shape[0]):       
        if j in q1 or j in q0:
            continue      
        else:   
            
            Up = Upper[j][:]
            Low = Lower[j][:]          
            Sum_1_D = 0.0
            Sum_2_D = sum(Low[Reach[j]])
            Phi_max[j][Reach_Sort[j][0]] = min(Low[Reach_Sort[j][0]] + 1 - Sum_2_D, Up[Reach_Sort[j][0]]) 
                             
            for i in range(1, len(Reach_Sort[j])):  
                               
                Sum_1_D = Sum_1_D + Phi_max[j][Reach_Sort[j][i-1]]               
                if Sum_1_D >= 1:
                    break     
                                       
                Sum_2_D = Sum_2_D - Low[Reach_Sort[j][i-1]]                                     
                Phi_max[j][Reach_Sort[j][i]] = min(Low[Reach_Sort[j][i]] + 1 - (Sum_1_D+Sum_2_D), Up[Reach_Sort[j][i]])
             
    
    
    return Phi_max





def Update_Ordering_Upper(State_Space, Q0, Q1, Int):
         
    Descending_Order = []
    First_State = 0 
    
    


    for k in range(State_Space.shape[0]):
              
        if k not in (Q0 + Q1):
            
            if First_State == 0:
                
                Descending_Order.append(k)
                First_State = 1
                
            else:
                
                for l in range(len(Descending_Order)):
                    
                    if (Int[Descending_Order[l]][0] < Int[k][0]):
                        
                        Descending_Order.insert(l, k)
                        break
                    
                    if l == len(Descending_Order) - 1:
                        Descending_Order.append(k)                        
                    
                       
    for k in range(State_Space.shape[0]):
                                        
        if k in Q1:
            Descending_Order.insert(0,k)
            
        if k in Q0:
            Descending_Order.append(k)
            
    return Descending_Order





def Reachability_Lower(IA_l, IA_u, Q1, Q0, Num_States, Automata_size, Reach, Init):
    
    
    Ascending_Order = []
    Index_Vector = np.zeros((IA_l.shape[0],1))  
    
    for k in range(IA_l.shape[0]):
                                
        if k in Q1:
            
            Index_Vector[k,0] = 1.0
            Ascending_Order.append(k)
           
        elif k not in Q0:
            
            Index_Vector[k,0] = 0.0
            Ascending_Order.insert(0,k)

    for k in range(len(Q0)): 
        Ascending_Order.insert(0,Q0[k])
        

    d = {k:v for v,k in enumerate(Ascending_Order)} 
    Sort_Reach = []

    for i in range(len(Reach)):
        Reach[i].sort(key=d.get)
        Sort_Reach.append(Reach[i])        
        
                
    Phi_Min = Phi_Computation_Lower(IA_u, IA_l, Ascending_Order, Q1, Q0, Reach, Sort_Reach)
    Steps_Low = np.dot(Phi_Min, Index_Vector)
    
    for i in range(len(Q1)):    
        Steps_Low[Q1[i]][0] = 1.0
    for i in range(len(Q0)):
        Steps_Low[Q0[i]][0] = 0.0

    
    Success_Intervals = []
         
    for i in range(IA_l.shape[0]):       
        Success_Intervals.append(Steps_Low[i][0])
              
    Terminate_Check = 0
    Convergence_threshold = 0.000001
    Previous_Max_Difference = 1
      
    while Terminate_Check == 0:
                   
        Previous_List = copy.copy(Ascending_Order)
        
        

        for i in range(len(Q0)):
            Success_Intervals[Q0[i]] = 0.0
        
        for i in range(len(Q1)):
            Success_Intervals[Q1[i]] = 1.0
       
        Ascending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Ascending_Order = list(Ascending_Order[(Success_Array).argsort()]) 
        
        d = {k:v for v,k in enumerate(Ascending_Order)} 
        Sort_Reach = []

        for i in range(len(Reach)):
            Reach[i].sort(key=d.get)
            Sort_Reach.append(Reach[i]) 
        
        if Previous_List != Ascending_Order:
            Phi_Min = Phi_Computation_Lower(IA_u, IA_l, Ascending_Order, Q1, Q0, Reach, Sort_Reach)
        Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Min), Steps_Low)
        
        for i in range(len(Q1)):   
            Steps_Low[Q1[i]][0] = 1.0
  
        for i in range(len(Q0)):
            Steps_Low[Q0[i]][0] = 0.0 
 

              
           
        Max_Difference = 0
        
               
        for i in range(IA_l.shape[0]):
           
                       
            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Steps_Low[i][0]))        
    
    
            Success_Intervals[i] = Steps_Low[i][0]
         
        if Max_Difference < Convergence_threshold:       
            Terminate_Check = 1
                   
    Bounds = []
    Prod_Bounds = []
    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
    for i in range(len(Success_Intervals)):
        Prod_Bounds.append(Success_Intervals[i])
    
       
    return (Bounds, Prod_Bounds, Phi_Min)




def Phi_Computation_Lower(Upper, Lower, Order_A, q1, q0, Reach, Reach_Sort):

    Phi_min = np.zeros((Upper.shape[0], Upper.shape[1]))


   
    for j in range(Upper.shape[0]):
        
        if j in q1 or j in q0:
            continue
        else:
    
            Up = Upper[j][:]
            Low = Lower[j][:]                 
            Sum_1_A = 0.0
            Sum_2_A = sum(Low[Reach[j]])
            Phi_min[j][Reach_Sort[j][0]] = min(Low[Reach_Sort[j][0]] + 1 - Sum_2_A, Up[Reach_Sort[j][0]])  
      
            for i in range(1, len(Reach_Sort[j])):
                             
                Sum_1_A = Sum_1_A + Phi_min[j][Reach_Sort[j][i-1]]
                if Sum_1_A >= 1:
                    break
                Sum_2_A = Sum_2_A - Low[Reach_Sort[j][i-1]]
                Phi_min[j][Reach_Sort[j][i]] = min(Low[Reach_Sort[j][i]] + 1 - (Sum_1_A+Sum_2_A), Up[Reach_Sort[j][i]])  
        
    return Phi_min






def Update_Ordering_Lower(State_Space, Q0, Q1, Int):
       
    Ascending_Order = []
    First_State = 0
    
    for k in range(State_Space.shape[0]):
              
        if k not in (Q0 + Q1):
            
            if First_State == 0:
                
                Ascending_Order.append(k)
                First_State = 1
                
            else:
                
                for l in range(len(Ascending_Order)):
                    
                    if (Int[Ascending_Order[-1-l]][0] < Int[k][0]):
                        
                        Ascending_Order.insert(-l, k)
                        break
                    
                    if l == len(Ascending_Order) - 1:
                        Ascending_Order.insert(0, k)
                                             
                                        
    for k in range(State_Space.shape[0]):
                                       
        
        if k in Q1:           
            Ascending_Order.append(k)

        elif k in Q0:          
            Ascending_Order.insert(0,k)
            
    return Ascending_Order




def SSCC(graph):
    
    #Search for all Strongly Connected Components in a Graph

    #set of visited vertices
    used = set()
    
    #call first depth-first search
    list_vector = [] #vertices in topological sorted order
    for vertex in range(len(graph)):
       if vertex not in used:
          (list_vector,used) = first_dfs(vertex, graph, used, list_vector)              
    list_vector.reverse()
    
    #preparation for calling second depth-first search
    graph_t = reverse_graph(graph)
    used = set()
    
    #call second depth-first search
    components= []
    list_components = [] #strong-connected components
    scc_quantity = 0 #quantity of strong-connected components 
    for vertex in list_vector:
        if vertex not in used:
            scc_quantity += 1
            list_components = []
            (list_components, used) = second_dfs(vertex, graph_t, list_components, list_vector, used)
#            print(list_components)
            components.append(list_components)
            
#    print(scc_quantity)
    
    return components, scc_quantity



def first_dfs(vertex, graph, used, list_vector):
    used.add(vertex)
    for v in range(len(graph)):   
        if graph[vertex][v] == 1 and v not in used:   
            (list_vector, used) = first_dfs(v, graph, used, list_vector)
    list_vector.append(vertex)
    return(list_vector, used)

    
def second_dfs(vertex, graph_t, list_components, list_vector, used):
    used.add(vertex)
    for v in list_vector:   
        if graph_t[vertex][v] == 1 and v not in used:   
            (list_components, used) = second_dfs(v, graph_t, list_components, list_vector, used)
    list_components.append(vertex)
    return(list_components, used)
    		                   
    
def reverse_graph(graph):
    graph_t = list(zip(*graph))
    return graph_t






def State_Space_Plot(Space, Y, N, M, Tag):
    
    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    if Tag == 0:
        fig = plt.figure('Verification against $\phi_1$')
        plt.rc('text', usetex='True')
        plt.rc('font', family='serif', serif= 'Helvetica')
        plt.title(r'Verification against $\phi$', fontsize=25)
        
    else:
        fig = plt.figure('Final State Space')
        plt.title(r'Final State Space', fontsize=25)
    

    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])

    ax=plt.gca()
    
    
    
    for i in range(Space.shape[0]):      
        if i in N:
            
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='red', edgecolor='k', linewidth = 0.1)
            
        elif i in Y:            
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='green', edgecolor='k', linewidth = 0.1)
        
        else:           
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='yellow', edgecolor='k', linewidth = 0.1)
        
        ax.add_artist(pol)
    
    ax1 = plt.gca()
    ax1.set_xlabel('$x_1$', fontsize=20)
    ax1.set_ylabel('$x_2$', fontsize=20)
    
      
    return 1




def Bounds_Tightening(Lower_Bound_Matrix, Upper_Bound_Matrix):
    
    
    for j in range(Lower_Bound_Matrix.shape[0]):
        Sum_Low = sum(Lower_Bound_Matrix[j][:])
        Sum_High = sum(Upper_Bound_Matrix[j][:])
        for i in range(Lower_Bound_Matrix.shape[1]):
            Res_Up = 1 - Sum_High - Upper_Bound_Matrix[j][i] - Lower_Bound_Matrix[j][i]
            Res_Down = 1 - Sum_Low -  Lower_Bound_Matrix[j][i] - Upper_Bound_Matrix[j][i]
            Lower_Bound_Matrix[j][i] = Lower_Bound_Matrix[j][i] + max(0, Res_Up)
            Upper_Bound_Matrix[j][i] = Upper_Bound_Matrix[j][i] + min(0, Res_Down)
            Sum_High = Sum_High + Res_Down
            Sum_Low = Sum_Low + Res_Up

    
    return Lower_Bound_Matrix, Upper_Bound_Matrix

def Raw_Refinement(State, Space): 
       
    New_St = []       
    for i in range(len(State)):

       a1 = Space[State[i]][1][0] - Space[State[i]][0][0]
       a2 = Space[State[i]][1][1] - Space[State[i]][0][1]
    
       if a1 > a2:
                               
           New_St.append([(Space[State[i]][0][0],Space[State[i]][0][1]),((Space[State[i]][1][0] + Space[State[i]][0][0])/2.0,Space[State[i]][1][1])])
           New_St.append([((Space[State[i]][1][0] + Space[State[i]][0][0])/2.0 , Space[State[i]][0][1]),(Space[State[i]][1][0],Space[State[i]][1][1])])
  
       else:
           
           New_St.append([(Space[State[i]][0][0] , (Space[State[i]][1][1]+Space[State[i]][0][1])/2.0),(Space[State[i]][1][0],Space[State[i]][1][1])])
           New_St.append([(Space[State[i]][0][0] , Space[State[i]][0][1]),(Space[State[i]][1][0],(Space[State[i]][1][1]+Space[State[i]][0][1])/2.0)])
   
    return New_St




def Random_Markov_Chain_IMC(Upp, Low):
    
    Random_Chain = np.zeros((Upp.shape[0], Upp.shape[0]))
    
    for i in range(Upp.shape[0]):
        Sum = 0.0
        for j in range(Upp.shape[0]):
            Random_Chain[i,j] = Low[i,j]
            Sum += Low[i,j]
            
        for j in range(Upp.shape[0]):
            if Sum >= 1.0: break
            Random_Chain[i,j] += min(Upp[i,j] - Low[i,j], 1.0-Sum)
            Sum += (Random_Chain[i,j] - Low[i,j])
            
    return Random_Chain 


