#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 03:19:06 2019

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
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from mpl_toolkits.mplot3d import Axes3D
import copy
import igraph


def Simulation(State, Label, Automata, Optimal_Policy, Init):
    
    Total_Number_of_Simulations = 5000
    Sim_Count = 0
    Number_of_Successes = 0

        
    while Sim_Count < Total_Number_of_Simulations:

        
        State_Counter = 0
        x0 = [0.15,-0.2]
        Time_Step_Total = 400
        Current_State = list(x0)
        j = 0;
        Tag = 0;
        
        
        while j < Time_Step_Total:
            
                      
            Next_State = list(Current_State)

            
            for i in range(len(State)): # i.e. 0 to 26. Therefore, a total length of 27.
                if(Current_State[0] <= State[i][1][0] and Current_State[1] <= State[i][1][1]  and Current_State[0] >= State[i][0][0] and Current_State[1] >= State[i][0][1]):
                    if j == 0:
                        p = Init[i] #Initial automaton state
                    State_no = i; #print State_no # Verify which partitioned state I am in.
                    State_Label = Label[State_no] # Verify which label that partitioned state corresponds to.
                    break
            
            #print p
            
            # Now looping through Automata to find corresponding automata_state via matching transition label 
            for i in range(len(Automata[p])): 
                for ii in range(len(Automata[p][i])):
                    if(Automata[p][i][ii] == State_Label):
                        Current_Auto_State = i; # Saving Current Automaton State.
                        break
    
            
            mode = Optimal_Policy[len(Automata)*State_no + Current_Auto_State] # Find the correct mode within policy
            

            X1 = norm(loc=0.0, scale=0.18)       

            
            if mode == 0:
            
                Next_State[0] = 6.0*(Current_State[0]**3)*Current_State[1] 
                Next_State[1] = min(max(-0.5, 0.3*Current_State[0]*Current_State[1]+ X1.rvs(1)), 0.5) # For truncation

            else:
                
                Next_State[0] = 7.0*(Current_State[0]**3)*Current_State[1] 
                Next_State[1] = min(max(-0.5, 0.2*Current_State[0]*Current_State[1]+ X1.rvs(1)), 0.5) # For truncation

    

            p = Current_Auto_State; # Updating placeholder index
            
            
            j= j+1;
                       
            Current_State = list(Next_State)
            
            if (Current_State[0] < -0.25 and Current_State[1] > 0.25) or (Current_State[0] > 0.25 and Current_State[1] < -0.25):
                break
            
            if j == 1 or j == 2:
                if (Current_State[0] > -0.25 and Current_State[0] < 0.0 and Current_State[1] > 0.25 and Current_State[1] < 0.5):
                    Tag = 1
                    break

            if (Current_State[0] > -0.5 and Current_State[0] < -0.25 and Current_State[1] > 0.0 and Current_State[1] < 0.25) or (Current_State[0] > 0.25 and Current_State[0] < 0.5 and Current_State[1] > -0.25 and Current_State[1] < 0.0):
                    Tag = 1
                    break                

          
        Sim_Count += 1
        print "Number of simulations completed"
        print Sim_Count
        if Tag == 1: #We consider the simulation to be a success if the system was in the accepting region for at least the last half of the simulation
            Number_of_Successes += 1

    
    return float(Number_of_Successes)/float(Total_Number_of_Simulations)    
    

def BMDP_Probability_Interval_Computation_Barrier(Target_Set, Domain, Reachable_States):
    

    Lower = np.array(np.zeros((2, Target_Set.shape[0],Target_Set.shape[0])))
    Upper = np.array(np.zeros((2,Target_Set.shape[0],Target_Set.shape[0])))
    Pre_States = [[[] for x in range(Target_Set.shape[0])] for y in range(2)]
    Is_Bridge_State = np.zeros((2,Target_Set.shape[0])) # Will not be used
    Bridge_Transitions = [[[] for x in range(Target_Set.shape[0])] for y in range(2)] #Will not be used
 
        
    
    eng = matlab.engine.start_matlab() #Start Matlab Engine
    

    
    for k in range(Target_Set.shape[0]):
        for j in range(Target_Set.shape[1]):
            for h in range(len(Reachable_States[k][j])):
                
    
                out = StringIO.StringIO()
                err = StringIO.StringIO()
                            
                Res = eng.Bounds_Computation_Synthesis(matlab.double(list(itertools.chain.from_iterable(Target_Set[j].tolist()))), matlab.double(list(itertools.chain.from_iterable(Target_Set[Reachable_States[k][j][h]].tolist()))), matlab.double(list([0.0, 1.0])), matlab.double(Domain), matlab.double([k]), stdout=out,stderr=err)
           
                H = Res[0][0]
                L = Res[0][1]
                if H > 0:
                    if L == 0:
                        Is_Bridge_State[k][j] = 1
                        Bridge_Transitions[k][j].append(Reachable_States[k][j][h])
                else:
                    Reachable_States[k][j].remove(Reachable_States[k][j][h])
                        
                Lower[k][j][h] = L
                Upper[k][j][h] = H
                
                
                         

    
    return Lower,Upper, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States




def Build_Product_BMDP(T_l, T_u, A, L, Acc, Reachable_States, Is_Bridge_State, Bridge_Transitions):
    

    
    Init = np.zeros((T_l.shape[1])) 
    Init = Init.astype(int)    
    
    Is_A = np.zeros(T_l.shape[1]*len(A))
    Is_N_A = np.zeros(T_l.shape[1]*len(A))
    Which_A = [[] for x in range(T_l.shape[1]*len(A))]
    Which_N_A = [[] for x in range(T_l.shape[1]*len(A))]
    
    New_Reachable_States = [[[] for x in range(T_l.shape[1]*len(A))] for y in range(T_l.shape[0])]
    New_Is_Bridge_State = np.zeros((T_l.shape[0], T_l.shape[1]*len(A)))
    New_Bridge_Transitions = [[[] for x in range(T_l.shape[1]*len(A))] for y in range(T_l.shape[0])]
    
    IA_l = np.zeros((T_l.shape[0],T_l.shape[1]*len(A), T_l.shape[2]*len(A)))
    IA_u = np.zeros((T_l.shape[0],T_l.shape[1]*len(A), T_l.shape[2]*len(A)))
    
    for x in range(len(Acc)): 
        for i in range(len(Acc[x][0])):
            for j in range(T_l.shape[1]):
                Is_N_A[len(A)*j + Acc[x][0][i]] = 1
                Which_N_A[len(A)*j + Acc[x][0][i]].append(x)
        
        for i in range(len(Acc[x][1])):
            for j in range(T_l.shape[1]):
                Is_A[len(A)*j + Acc[x][1][i]] = 1
                Which_A[len(A)*j + Acc[x][1][i]].append(x)            
                
    
    for y in range(T_l.shape[0]): 
        for i in range(T_l.shape[1]):
            for j in range(len(A)):
                for k in range(T_l.shape[2]):
                    for l in range(len(A)):
                        
                        if L[k] in A[j][l]:
                            
                            if y == 0 and j == 0: 
                                Init[k] = l
    
                            IA_l[y, len(A)*i+j, len(A)*k+l] = T_l[y,i,k]  
                            IA_u[y, len(A)*i+j, len(A)*k+l] = T_u[y,i,k] 
                            

                            
                            if T_u[y,i,k] > 0:
                                New_Reachable_States[y][len(A)*i+j].append(len(A)*k+l)
                                if T_l[y,i,k] == 0:
                                    New_Is_Bridge_State[y, len(A)*i+j] = 1 
                                    New_Bridge_Transitions[y][len(A)*i+j].append(len(A)*k+l)
                            
                        else:
                            IA_l[y, len(A)*i+j, len(A)*k+l] = 0.0
                            IA_u[y, len(A)*i+j, len(A)*k+l] = 0.0
    
                 

    Is_A = Is_A.astype(int)
    Is_N_A = Is_N_A.astype(int)
    New_Is_Bridge_State = New_Is_Bridge_State.astype(int)                         

    return (IA_l, IA_u, Is_A, Is_N_A, Which_A, Which_N_A, New_Reachable_States, New_Is_Bridge_State, New_Bridge_Transitions, Init) 







def Find_Greatest_Accepting_BSCCs(IA1_l, IA1_u, Is_Acc, Is_NAcc, Wh_Acc_Pair, Wh_NAcc_Pair, Al_Act_Pot, Al_Act_Perm, first, Reachable_States, Bridge_Transition, Is_Bridge_State, Acc, Potential_Policy, Permanent_Policy, Is_In_Permanent_Comp, List_Permanent_Acc_BSCC, Previous_A_BSCC):

    G = np.zeros((IA1_l.shape[1],IA1_l.shape[2]))

    
    if first == 1:
        Al_Act_Pot = list([])
        Al_Act_Perm = list([])
        for y in range(IA1_l.shape[1]):
            Al_Act_Pot.append(range(IA1_l.shape[0])) 
            Al_Act_Perm.append(range(IA1_l.shape[0]))
    
    
    for k in range(IA1_u.shape[0]): 
        for i in range(IA1_u.shape[1]): 
            for j in range(IA1_u.shape[2]):
                if IA1_u[k,i,j] > 0:  
                    G[i,j] = 1

    Counter_Status2 = 0 
    Counter_Status3 = 0 
    Which_Status2_BSCC = [] 
    Has_Found_BSCC_Status_2 = list([]) 
    List_Found_BSCC_Status_2 = list([]) 
    Original_SCC_Status_2 = list([]) 
    Which_Status3_BSCC = list([])
    Number_Duplicates2 = 0 
    Number_Duplicates3 = 0 
    Status2_Act = list([]) 
    Status3_Act = list([])
    List_Status3_Found = list([])

    if first == 0:
        Deleted_States = []
        Prev_A = set().union(*Previous_A_BSCC)
        Deleted_States.extend(list(set(range(G.shape[0])) - set(Prev_A)))
        
        Ind = list(set(Prev_A))
        Ind.sort()
        
        G = np.delete(np.array(G),Deleted_States,axis=0)
        G = np.delete(np.array(G),Deleted_States,axis=1)
        

    else:
        Ind = range(G.shape[0])
     
    
        
    first = 0 

   
    C,n = SSCC(G)     
    tag = 0; 
    m = 0 ;

    
    SCC_Status = [0]*n 
   
    G_Pot_Acc_BSCCs = list([]) 
    G_Per_Acc_BSCCs = list([]) 
    
    
    for i in range(len(List_Permanent_Acc_BSCC)): 
        for j in range(len(List_Permanent_Acc_BSCC[i])):
            G_Pot_Acc_BSCCs.append(List_Permanent_Acc_BSCC[i][j])
            G_Per_Acc_BSCCs.append(List_Permanent_Acc_BSCC[i][j])
    
    List_G_Pot = [] 
    Is_In_Potential_Acc_BSCC = np.zeros(IA1_l.shape[1]) 
    Which_Potential_Acc_BSCC = np.zeros(IA1_l.shape[1]) 
    Which_Potential_Acc_BSCC.astype(int)
    Is_In_Potential_Acc_BSCC.astype(int)
    Bridge_Potential_Accepting = [] 
    Maybe_Permanent = [] 
    
    
    
    while tag == 0:
        
        if len(C) == 0:
            break
        
        
        skip = 1 
        SCC = C[m];
        

        

        Orig_SCC = []
        for k in range(len(SCC)):
            Orig_SCC.append(Ind[SCC[k]]) 
        BSCC = 1
    
        
        if len(Has_Found_BSCC_Status_2) != 0:
            if SCC_Status[m] == 2 and Has_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]] == 1:
                Counter_Status2 += 1
                if m < (len(C)-1): 
                    m += 1                    
                    continue 
                else:                   
                    break 



        for l in range(len(Orig_SCC)):
            if Is_Acc[Orig_SCC[l]] == 1: 
                skip = 0
                break

        if skip == 1: 
            if SCC_Status[m] == 0: 
                for i in range(len(SCC)):
                    Al_Act_Pot[Ind[SCC[i]]] = list([])
                    Al_Act_Perm[Ind[SCC[i]]] = list([])            
            if m < (len(C)-1): 
                m += 1 
                continue 
            else: 
                break     

            
        
        
        Leak = list([])
        Check_Tag = 1
        Reach_in_R = [[[] for y in range(IA1_u.shape[0])] for x in range(len(Orig_SCC))] # Reach_in_R contains all the reachable non-leaky states inside the SCC, with respect to state i.
        Pre = [[[] for y in range(IA1_u.shape[0])] for x in range(len(Orig_SCC))] # Creating list of list of lists, to account for mode, state, transitions. Modes are nested inside state.
        All_Leaks = list([])
        Check_Orig_SCC = np.zeros(len(Orig_SCC), dtype=int)
        

            
            

        while (len(Leak) != 0 or Check_Tag == 1):
                       
                        
            if SCC_Status[m] == 0:
                                                              
                ind_leak = []
                Leak = []
                      
                for i in range(len(Orig_SCC)): 
                    if Check_Orig_SCC[i] == -1 :
                        continue 
                    tag_m = 0
                    
                    for k in range(len(Al_Act_Pot[Orig_SCC[i]])): 
        
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) 
                        Diff_List1 = list(set(Reachable_States[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) 
                        Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]))  
                        if Check_Tag == 1: 

                            Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]].extend(list(set(Reachable_States[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - set(Diff_List1)))
                            for j in range(len(Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]])):
                                Pre[Orig_SCC.index(Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]][j])][Al_Act_Pot[Orig_SCC[i]][k-tag_m]].append(Orig_SCC[i]) 
                       
    
                        if (len(Diff_List2) != 0) or (sum(IA1_u[Al_Act_Pot[Orig_SCC[i]][k-tag_m], Orig_SCC[i], Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]]])<1) : 
                            
                            Al_Act_Perm[Orig_SCC[i]].remove(Al_Act_Pot[Orig_SCC[i]][k-tag_m]) 
                            Al_Act_Pot[Orig_SCC[i]].remove(Al_Act_Pot[Orig_SCC[i]][k-tag_m]) 
                            tag_m += 1 
                            BSCC = 0 
                            
                    if len(Al_Act_Pot[Orig_SCC[i]]) == 0:                        
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                        
                if len(Leak) != 0: 
                    All_Leaks.extend(Leak) 
                    BSCC = 0 
                    for i in range(len(Leak)): 
                        Check_Orig_SCC[ind_leak[i]] = -1 
                        for j in range(len(Pre[ind_leak[i]])): 
                            for k in range(len(Pre[ind_leak[i]][j])): 
                                Reach_in_R[Orig_SCC.index(Pre[ind_leak[i]][j][k])][j].remove(Leak[i]) 
                Check_Tag = 0  
 
            if SCC_Status[m] == 1:
                
                

 
                ind_leak = []
                Leak = []                     
                for i in range(len(Orig_SCC)): 
                    if Check_Orig_SCC[i] == -1 :
                        continue 
                    tag_m = 0
                     


                    for k in range(len(Al_Act_Perm[Orig_SCC[i]])):
              
                
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) 

                        Diff_List1 = list(set(Reachable_States[Al_Act_Perm[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) 

                        if (len(Diff_List1) != 0):                            
                            Al_Act_Perm[Orig_SCC[i]].remove(Al_Act_Perm[Orig_SCC[i]][k-tag_m]) 
                            tag_m += 1 
                            BSCC = 0 
                                                    
                    if len(Al_Act_Perm[Orig_SCC[i]]) == 0:                     
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                if len(Leak) != 0: 
                    All_Leaks.extend(Leak) 
                    BSCC = 0 
                    for i in range(len(Leak)): 
                        Check_Orig_SCC[ind_leak[i]] = -1 
                Check_Tag = 0  


            if SCC_Status[m] == 2:
                
                                                             
                ind_leak = []
                Leak = []                      
                for i in range(len(Orig_SCC)): 
                    if Check_Orig_SCC[i] == -1 :
                        continue 
                    tag_m = 0
                    
                    for k in range(len(Status2_Act[Counter_Status2][Orig_SCC[i]])): 
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) 
                        Diff_List1 = list(set(Reachable_States[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) 
                        Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]))  
                        if Check_Tag == 1: 

                            Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]].extend(list(set(Reachable_States[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - set(Diff_List1)))
                            for j in range(len(Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]])):
                                Pre[Orig_SCC.index(Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][j])][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]].append(Orig_SCC[i]) 
                       
    
                        if (len(Diff_List2) != 0) or (sum(IA1_u[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m], Orig_SCC[i], Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]]])<1) : 
                            
                            Status2_Act[Counter_Status2][Orig_SCC[i]].remove(Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]) 
                            tag_m += 1 
                            BSCC = 0 
                            
                    if len(Status2_Act[Counter_Status2][Orig_SCC[i]]) == 0:               
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                        
                if len(Leak) != 0: 
                    All_Leaks.extend(Leak) 
                    BSCC = 0 
                    for i in range(len(Leak)): 
                        Check_Orig_SCC[ind_leak[i]] = -1 
                        for j in range(len(Pre[ind_leak[i]])): 
                            for k in range(len(Pre[ind_leak[i]][j])): 
                                Reach_in_R[Orig_SCC.index(Pre[ind_leak[i]][j][k])][j].remove(Leak[i]) 
                Check_Tag = 0  
 

            if SCC_Status[m] == 3:
                
                                
                ind_leak = []
                Leak = []                     
                for i in range(len(Orig_SCC)): 
                    if Check_Orig_SCC[i] == -1 :
                        continue 
                    tag_m = 0
            
                    for k in range(len(Status3_Act[Counter_Status3][Orig_SCC[i]])): 
                          
                                                   
                                        
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks)          
                        
                        
                        Diff_List1 = list(set(Reachable_States[Status3_Act[Counter_Status3][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks)
    

    
                        if (len(Diff_List1) != 0):                          
                            Status3_Act[Counter_Status3][Orig_SCC[i]].remove(Status3_Act[Counter_Status3][Orig_SCC[i]][k-tag_m]) 
                            tag_m += 1 
                            BSCC = 0 
                            
                    if len(Status3_Act[Counter_Status3][Orig_SCC[i]]) == 0:                  
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                if len(Leak) != 0: 
                    All_Leaks.extend(Leak) 
                    BSCC = 0 
                    for i in range(len(Leak)): 
                        Check_Orig_SCC[ind_leak[i]] = -1 
                Check_Tag = 0  



               
        if BSCC == 0: 
            
            
            SCC = list(set(Orig_SCC) - set(All_Leaks))
            for k in range(len(SCC)):
                SCC[k] = Ind.index(SCC[k])
            
            if SCC_Status[m] == 0:  
                
                
                                
                if len(SCC) != 0: 
                    SCC = sorted(SCC, key=int)               
                    New_G = np.zeros((len(SCC), len(SCC)))
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1
                        
                    C_new, n_new = SSCC(New_G)               
                    for j in range(len(C_new)):
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]]
                        C.append(C_new[j]) 
                        SCC_Status.append(0)

            if SCC_Status[m] == 1: 
                
                
                
                if len(SCC) != 0: 
                    SCC = sorted(SCC, key=int)                
                    New_G = np.zeros((len(SCC), len(SCC)))
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Perm[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Perm[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:                                        
                                    New_G[i,j] = 1
                    C_new, n_new = SSCC(New_G)                
                    

                    
                    for j in range(len(C_new)):
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]]
                        C.append(C_new[j]) 
                        SCC_Status.append(1)

            if SCC_Status[m] == 2:
                                                
                            
                if len(SCC) != 0: 
                    Duplicate_Actions = copy.deepcopy(Status2_Act[Counter_Status2])
                    SCC = sorted(SCC, key=int)                
                    New_G = np.zeros((len(SCC), len(SCC)))
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1

                    C_new, n_new = SSCC(New_G)                
                    for j in range(len(C_new)):
                        Status2_Act.append(Duplicate_Actions)
                        Which_Status2_BSCC.append(Which_Status2_BSCC[Counter_Status2])
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] 
                        C.append(C_new[j]) 
                        SCC_Status.append(2)                                        
                Counter_Status2 += 1

            if SCC_Status[m] == 3: 
                
                
                if len(SCC) != 0: #
                    Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                    SCC = sorted(SCC, key=int)               
                    New_G = np.zeros((len(SCC), len(SCC)))
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1

                    C_new, n_new = SSCC(New_G)                
                    for j in range(len(C_new)):
                        Status3_Act.append(Duplicate_Actions)
                        Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] 
                        C.append(C_new[j]) 
                        SCC_Status.append(3)                                        
                Counter_Status3 += 1                
            
        else: 
            
            
            Bridge_States = []           
            if SCC_Status[m] == 0:
                            

                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: 
                        acc_states.append(SCC[j])
                        indices = [] 
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): 
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) 
                        ind_acc.append(indices)

                    if Is_NAcc[Ind[SCC[j]]] == 1:
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                    for i in range(len(Al_Act_Pot[Ind[SCC[j]]])): 
                        if Is_Bridge_State[Al_Act_Pot[Ind[SCC[j]]][i]][Ind[SCC[j]]] == 1:                                                                   
                            Diff_List = np.setdiff1d(Reachable_States[Al_Act_Pot[Ind[SCC[j]]][i]][Ind[SCC[j]]], Orig_SCC)   
                            if len(Diff_List) != 0: 
                                Inevitable = 0  
                            Bridge_States.append(Ind[SCC[j]]) 

                Acc_Tag = 0
                Accept = [] 
                                                   
                if len(non_acc_states) == 0: 
                    Acc_Tag = 1 
                    for j in range(len(acc_states)): 
                        Accept.append(acc_states[j])
                
                else:
                    
                    Non_Accept_Remove = [[] for x in range(len(Acc))]                          
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
                    SCC.sort()
                    Accept.sort()
                    Potential_Policy_BSCC = np.zeros(len(SCC))
                    Permanent_Policy_BSCC = np.zeros(len(SCC))  
                    for i in range(len(Accept)):  
                        Act1 = Al_Act_Perm[Ind[Accept[i]]][0]
                        Act2 = Al_Act_Pot[Ind[Accept[i]]][0] 
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[Accept[i]] = Act1
                        Potential_Policy_BSCC[Accept[i]] = Act2
                           
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                                if IA1_u_BSCC[Al_Act_Pot[Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Al_Act_Pot[Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Al_Act_Pot[Indices[i]])  
                    
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Potential_Policy_BSCC, Dum) = Maximize_Upper_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Potential_Policy_BSCC, BSCC_Allowed_Actions) 
                    for i in range(len(SCC)):
                        Potential_Policy[Ind[SCC[i]]] = Potential_Policy_BSCC[i]
                        G_Pot_Acc_BSCCs.append(Ind[SCC[i]])
                    

                    Maybe_Permanent.append(SCC)
                    
                                       
                                       
                    if Inevitable == 1: 


                          
                        
                        (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions) 
                        Bad_States = []
                        for i in range(len(Dummy_Low_Bounds)):
                            if Dummy_Low_Bounds[i] == 0: 
                                Bad_States.append(SCC[i])
                        if len(Bad_States) == 0:                            
                            List_Permanent_Acc_BSCC.append([])
                            for i in range(len(SCC)):                                
                                Permanent_Policy[Ind[SCC[i]]] = Permanent_Policy_BSCC[i]
                                Potential_Policy[Ind[SCC[i]]] = Permanent_Policy[Ind[SCC[i]]] 
                                G_Per_Acc_BSCCs.append(Ind[SCC[i]])
                                G_Pot_Acc_BSCCs.append(Ind[SCC[i]])
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                                List_Permanent_Acc_BSCC[-1].append(Ind[SCC[i]])
                            Maybe_Permanent.pop()
                        else:
                                  

                            SCC_New = list(set(SCC) - set(Bad_States)) 
                            SCC_New.sort()        
                            if len(SCC_New) != 0:
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                    
                                C_new, n_new = SSCC(New_G) 
                              
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] 
                                    C.append(C_new[j]) 
                                    SCC_Status.append(1)
                          

                            SCC_New = list(Bad_States)
                            SCC_New.sort()        
                            if len(SCC_New) != 0: 
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                    
                                C_new, n_new = SSCC(New_G) 
                              
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]]
                                    C.append(C_new[j]) 
                                    SCC_Status.append(1)                            
                             
                                                  
                    else:
                        C.append(SCC)
                        SCC_Status.append(1)
                        
                                                    
                else: 
                    
                    Check_Tag2 = 0
                    Count_Duplicates = 0
                    
                    
                    for j in range(len(Non_Accept_Remove)):
                        if len(Non_Accept_Remove[j]) != 0:
                            Count_Duplicates += 1
                    
                    for j in range(len(Non_Accept_Remove)): 
                        if len(Non_Accept_Remove[j]) != 0:
                    
                            if Check_Tag2 == 0 and Count_Duplicates > 1:
                                Duplicate_Actions = copy.deepcopy(Al_Act_Pot)
                                Has_Found_BSCC_Status_2.append(0)
                                List_Found_BSCC_Status_2.append([])
                                Original_SCC_Status_2.append(SCC)
                                Check_Tag2 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) 
                            SCC_New.sort()

                            if len(SCC_New) != 0: 
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                    
                                C_new, n_new = SSCC(New_G) 
                                
                                for j in range(len(C_new)):
                                    if Count_Duplicates > 1:
                                        Status2_Act.append(Duplicate_Actions)
                                        Which_Status2_BSCC.append(Number_Duplicates2)                                    
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] 
                                    
                                    C.append(C_new[j]) 
                                   
                                    if Count_Duplicates > 1:
                                        SCC_Status.append(2)
                                    else:    
                                        SCC_Status.append(0)
                                        
                    if Check_Tag2 == 1 and Count_Duplicates > 1:
                        Number_Duplicates2 += 1

                        
            elif SCC_Status[m] == 1: 
                

                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                
                               
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: 
                        acc_states.append(SCC[j])
                        indices = [] 
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): 
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) 
                        ind_acc.append(indices) 

                    if Is_NAcc[Ind[SCC[j]]] == 1: 
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                Acc_Tag = 0
                Accept = [] 
                                                    
                if len(non_acc_states) == 0: 
                    Acc_Tag = 1 
                    for j in range(len(acc_states)): 
                        Accept.append(acc_states[j])
                
                else:                                        
                  
                    Non_Accept_Remove = [[] for x in range(len(Acc))] 
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
                   
                    
                    
                    SCC.sort()
                    Accept.sort()
                    Permanent_Policy_BSCC = np.zeros(len(SCC))
                    for i in range(len(Accept)): 
                        Act = Al_Act_Perm[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[i] = Act
                        
                                      
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    
                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Al_Act_Perm[Ind[SCC[i]]])):
                                if IA1_u_BSCC[Al_Act_Perm[Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Al_Act_Perm[Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Al_Act_Perm[Indices[i]])  
                    
                    
                                     
                    (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions) 
                    Bad_States = []
                    for i in range(len(Dummy_Low_Bounds)):
                        if Dummy_Low_Bounds[i] == 0: 
                            Bad_States.append(SCC[i])


                    if len(Bad_States) == 0:                    
                        if SCC not in List_Permanent_Acc_BSCC:
                            List_Permanent_Acc_BSCC.append([])
                            for i in range(len(SCC)):
                                Permanent_Policy[Ind[SCC[i]]] = Permanent_Policy_BSCC[i]
                                Potential_Policy[Ind[SCC[i]]] = Permanent_Policy[Ind[SCC[i]]] 
                                G_Per_Acc_BSCCs.append(Ind[SCC[i]])
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                                List_Permanent_Acc_BSCC[-1].append(Ind[SCC[i]])
                    else:                      
                        SCC_New = list(set(SCC) - set(Bad_States)) 
                        SCC_New.sort()          
                        if len(SCC_New) != 0: 
                            SCC = sorted(SCC, key=int) 
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))
                            for i in range(len(SCC_New)):
                                for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
                                
                            C_new, n_new = SSCC(New_G) 
                            for j in range(len(C_new)):
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] 
                                C.append(C_new[j]) 
                                SCC_Status.append(1)
                         
                        SCC_New = list(Bad_States) 
                        SCC_New.sort()          
                        if len(SCC_New) != 0: 
                            SCC = sorted(SCC, key=int) 
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))
                            for i in range(len(SCC_New)):
                                for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
                                
                            C_new, n_new = SSCC(New_G) 
                            for j in range(len(C_new)):
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] 
                                C.append(C_new[j]) 
                                SCC_Status.append(1)


                else:
                    Check_Tag3 = 0
                    
                    
                    Count_Duplicates = 0                   
                    for j in range(len(Non_Accept_Remove)):
                        if len(Non_Accept_Remove[j]) != 0:
                            Count_Duplicates += 1
                                                               
                            
                    for j in range(len(Non_Accept_Remove)): 
                        if len(Non_Accept_Remove[j]) != 0: 
                            if Check_Tag3 == 0 and Count_Duplicates > 1:
                                Duplicate_Actions = copy.deepcopy(Al_Act_Pot)
                                Check_Tag3 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) 
                            SCC_New.sort()
                            if len(SCC_New) != 0: 
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                  
                                C_new, n_new = SSCC(New_G) 
                                for j in range(len(C_new)):

                                    if Count_Duplicates > 1:
                                        Status3_Act.append(Duplicate_Actions)
                                        Which_Status3_BSCC.append(Number_Duplicates3)
                                        List_Status3_Found.append([])
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] 
                                    C.append(C_new[j]) 
                                    
                                    if Count_Duplicates > 1:
                                        
                                        SCC_Status.append(3)
                                    else:
                                        SCC_Status.append(1)
#                   
                    if Check_Tag3 == 1 and Count_Duplicates > 1:
                        Number_Duplicates3 += 1
                    
                        
            elif SCC_Status[m] == 2:
                
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                       
                
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: 
                        acc_states.append(SCC[j])
                        indices = [] 
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): 
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) 
                        ind_acc.append(indices) 

                    if Is_NAcc[Ind[SCC[j]]] == 1: 
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                    for i in range(len(Status2_Act[Counter_Status2][Ind[SCC[j]]])): 
                        if Is_Bridge_State[Status2_Act[Counter_Status2][Ind[SCC[j]]][i]][Ind[SCC[j]]] == 1: 
                            Diff_List = np.setdiff1d(Reachable_States[Status2_Act[Counter_Status2][Ind[SCC[j]]][i]][Ind[SCC[j]]], Orig_SCC) 
                            if len(Diff_List) != 0: 
                                Inevitable = 0  
                            Bridge_States.append(Ind[SCC[j]]) 

                Acc_Tag = 0
                Accept = [] 
                                                   
                if len(non_acc_states) == 0: 
                    Acc_Tag = 1 
                    for j in range(len(acc_states)): 
                        Accept.append(acc_states[j])
                
                else:
                                               
                    Non_Accept_Remove = [[] for x in range(len(Acc))] 
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
                    
                    Has_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]] = 1
                    SCC.sort()
                    Accept.sort()
                    Potential_Policy_BSCC = np.zeros(len(SCC)) 
                    for i in range(len(Accept)):  
                        Act = Al_Act_Pot[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Potential_Policy_BSCC[i] = Act
                                      
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Status2_Act[Counter_Status2][Ind[SCC[i]]])):
                                if IA1_u_BSCC[Status2_Act[Counter_Status2][Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Status2_Act[Counter_Status2][Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Status2_Act[Counter_Status2][Indices[i]])  
                    
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Potential_Policy_BSCC, Dum) = Maximize_Upper_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Potential_Policy_BSCC, BSCC_Allowed_Actions) 
                    for i in range(len(SCC)):
                        Potential_Policy[Ind[SCC[i]]] = Potential_Policy_BSCC[i]
                        List_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]].append(Ind[SCC[i]])
                         
                    C.append(Original_SCC_Status_2[Which_Status2_BSCC[Counter_Status2]]) 
                    
                    SCC_Status.append(1)                

                else: 
                    
                    
                    Check_Tag2 = 0
                    
                    for j in range(len(Non_Accept_Remove)): 
                        if len(Non_Accept_Remove[j]) != 0:
                            if Check_Tag2 == 0:
                                Duplicate_Actions = copy.deepcopy(Status2_Act[Counter_Status2])
                                Check_Tag2 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) 
                            SCC_New.sort()

                            if len(SCC_New) != 0: 
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                    
                                C_new, n_new = SSCC(New_G) 
                                
                                for j in range(len(C_new)):
                                    Status2_Act.append(Duplicate_Actions)
                                    Which_Status2_BSCC.append(Which_Status2_BSCC[Counter_Status2])                                   
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] 
                                    
                                    C.append(C_new[j]) 
                                    SCC_Status.append(2)
                        

                                                                            
                Counter_Status2 +=1


            elif SCC_Status[m] == 3:  

                
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: 
                        acc_states.append(SCC[j])
                        indices = [] 
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): 
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) 
                        ind_acc.append(indices) 

                    if Is_NAcc[Ind[SCC[j]]] == 1: 
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                Acc_Tag = 0
                Accept = [] 
                                                    
                if len(non_acc_states) == 0: 
                    Acc_Tag = 1 
                    for j in range(len(acc_states)): 
                        Accept.append(acc_states[j])
                
                else:                                        
                  
                    Non_Accept_Remove = [[] for x in range(len(Acc))] 
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
                
                    SCC.sort()
                    Accept.sort()
                    Permanent_Policy_BSCC = np.zeros(len(SCC))  
                    for i in range(len(Accept)):   
                        Act = Al_Act_Perm[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[i] = Act
                          
                                    
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    
                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Status3_Act[Counter_Status3][Ind[SCC[i]]])):
                                if IA1_u_BSCC[Status3_Act[Counter_Status3][Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Status3_Act[Counter_Status3][Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Status3_Act[Counter_Status3][Indices[i]])  
                    
                                     
                    (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions) 
                    Bad_States = []
                    for i in range(len(Accept)):
                        Permanent_Policy_BSCC[Accept[i]] = Status3_Act[Counter_Status3][Ind[SCC[Accept[i]]]][0]
                        
                    for i in range(len(Dummy_Low_Bounds)):
                        if Dummy_Low_Bounds[i] == 0: 
                            Bad_States.append(SCC[i])
                    
                    if len(Bad_States) == 0:
                        Existing_Lists = []
                        for i in range(len(List_Status3_Found[Which_Status3_BSCC[Counter_Status3]])):
                            Existing_Lists.append(List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][i][0])
                        
                        if SCC not in Existing_Lists:
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]].append([])
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1].append([]) 
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1].append([])
                            for i in range(len(SCC)):                            
                                List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1][0].append(Ind[SCC[i]])
                                List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1][1].append(Permanent_Policy_BSCC[i])
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                    else:                      
                        SCC_New = list(set(SCC) - set(Bad_States)) 
                        SCC_New.sort()          
                        if len(SCC_New) != 0: 
                            Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                            SCC = sorted(SCC, key=int)  
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))
                            for i in range(len(SCC_New)):
                                for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
                                
                            C_new, n_new = SSCC(New_G) 
                            for j in range(len(C_new)):
                                Status3_Act.append(Duplicate_Actions)
                                Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])                                
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] 
                                C.append(C_new[j]) 
                                SCC_Status.append(3)
  

                        SCC_New = list(Bad_States)
                        SCC_New.sort()          
                        if len(SCC_New) != 0: 
                            Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                            SCC = sorted(SCC, key=int) 
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))
                            for i in range(len(SCC_New)):
                                for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
                                
                            C_new, n_new = SSCC(New_G) 
                            for j in range(len(C_new)):
                                Status3_Act.append(Duplicate_Actions)
                                Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])                                
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] 
                                C.append(C_new[j]) 
                                SCC_Status.append(3)                       

                else: 

                    
                    Check_Tag3 = 0 
                    for j in range(len(Non_Accept_Remove)): 
                        if len(Non_Accept_Remove[j]) != 0: 
                            if Check_Tag3 == 0:
                                Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                                Check_Tag3 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) 
                            SCC_New.sort()
                            if len(SCC_New) != 0: 
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))
                                for i in range(len(SCC_New)):
                                    for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                  
                                C_new, n_new = SSCC(New_G) 
                                for j in range(len(C_new)):
                                    Status3_Act.append(Duplicate_Actions)
                                    Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] 
                                    C.append(C_new[j]) 
                                    SCC_Status.append(3)
#                   
                
                Counter_Status3 +=1                 
                    
        m +=1
        if m == len(C): tag = 1
      
    for i in range(len(Maybe_Permanent)): 
        List_Potential_States = []
        List_Bridge_States = []
        BSCC_Converted_Indices = []
        for j in range(len(Maybe_Permanent[i])):            
            if Is_In_Permanent_Comp[Ind[Maybe_Permanent[i][j]]] == 0:
                BSCC_Converted_Indices.append(Ind[Maybe_Permanent[i][j]])
                List_Potential_States.append(Ind[Maybe_Permanent[i][j]])
                Which_Potential_Acc_BSCC[Ind[Maybe_Permanent[i][j]]] = len(List_G_Pot)
                Is_In_Potential_Acc_BSCC[Ind[Maybe_Permanent[i][j]]] = 1
                if Is_Bridge_State[Potential_Policy[Ind[Maybe_Permanent[i][j]]]][Ind[Maybe_Permanent[i][j]]] == 1:
                    List_Bridge_States.append(Ind[Maybe_Permanent[i][j]])
        if len(List_Potential_States) != 0: 
            List_G_Pot.append(BSCC_Converted_Indices)
            Bridge_Potential_Accepting.append(List_Bridge_States)


            
    for i in range(Number_Duplicates2): 
        if Has_Found_BSCC_Status_2[i] == 1: 
            Non_Permanent_States = []
            for j in range(len(Original_SCC_Status_2[i])):
                G_Pot_Acc_BSCCs.append(Ind[Original_SCC_Status_2[i][j]])
                if Is_In_Permanent_Comp[Ind[Original_SCC_Status_2[i][j]]] == 0:
                    Non_Permanent_States.append(Ind[Original_SCC_Status_2[i][j]])
                    Is_In_Potential_Acc_BSCC[Ind[Original_SCC_Status_2[i][j]]] = 1
                    Which_Potential_Acc_BSCC[Ind[Original_SCC_Status_2[i][j]]] = len(List_G_Pot)
            List_G_Pot.append(Non_Permanent_States) 
            Remaining_States = list(set(Original_SCC_Status_2[i]) - set(List_Found_BSCC_Status_2[i]))
            for j in range(len(Remaining_States)):
                Potential_Policy[Ind[Remaining_States[i]]] = Al_Act_Pot[Ind[Remaining_States[i]]][0] 
            List_Bridge_States = []
            for j in range(len(Non_Permanent_States)):          
                if Is_Bridge_State[Potential_Policy[Non_Permanent_States[j]]][Non_Permanent_States[j]] == 1: 
                    List_Bridge_States.append(Non_Permanent_States[j])
            Bridge_Potential_Accepting.append(List_Bridge_States) 


                       
    for i in range(Number_Duplicates3):
        if (len(List_Status3_Found[i])!= 0):
            Graph = np.zeros((len(List_Status3_Found[i]),len(List_Status3_Found[i])))
            for j in range(len(List_Status3_Found[i])): 
                Graph[j,j] = 1
                for k in range(j+1, len(List_Status3_Found[i])):
                                     
                    if (set(List_Status3_Found[i][j][0]).intersection(set(List_Status3_Found[i][k][0]))) != 0:
                        Graph[j,k] = 1
                        Graph[k,j] = 1
            
            

            Comp_Graph = csr_matrix(Graph)
            Num_Comp, labels =  connected_components(csgraph=Comp_Graph, directed=False, return_labels=True)            
            C = [[] for x in range(Num_Comp)]
    
            for k in range(len(labels)):
                C[labels[i]].append(i)
                
            for k in range(len(C)):
        
                Component = []
                if len(C[k]) == 1:   
                    for l in range(len(List_Status3_Found[i][C[k][0]][0])):

                        Permanent_Policy[Ind[List_Status3_Found[i][C[k][0]][0][l]]] = List_Status3_Found[i][C[k][0]][1][l]
                        Component.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                    
                    List_Permanent_Acc_BSCC.append(Component)
  
                      
                else:
                                        
                    States_To_Reach = []
                    for l in range(len(List_Status3_Found[i][C[k][0]][0])):
                        Permanent_Policy[Ind[List_Status3_Found[i][C[k][0]][0][l]]] = List_Status3_Found[i][C[k][0]][1][l]
                        States_To_Reach.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                        Component.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                   
                    States_For_Reachability = []
                    for l in range(1, len(C[k])):
                        for m in range(len(List_Status3_Found[i][C[k][l]][0])):
                            if Ind[List_Status3_Found[i][C[k][l]][0][m]] not in States_To_Reach:
                                States_For_Reachability.append(Ind[List_Status3_Found[i][C[k][l]][0][m]])
                                Component.append(Ind[List_Status3_Found[i][C[k][l]][0][m]])


                    Component.sort()
                    Policy_BSCC = np.zeros(len(Component))                   
                    BSCC_Reachable_States = []
                    Indices = []
                    for y in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for x in range(len(Component)):
                            if y == 0:
                                Indices.append(Component[x])
                            BSCC_Reachable_States[-1].append([])
                            
                    
                    Target = []
                    for y in range(len(States_To_Reach)):
                        Target.append(Indices.index(States_To_Reach[y]))

                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for y in range(len(Component)):
                        for x in range(len(Component)):
                            for l in range(len(Al_Act_Perm[Component[y]])):
                                if IA1_u_BSCC[Al_Act_Perm[Component[y]][l], y,x] > 0:
                                    BSCC_Reachable_States[Al_Act_Perm[Component[y]][l]][y].append(x)                                        
                    BSCC_Allowed_Actions = []
                    for y in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Al_Act_Perm[Indices[y]])  
                    
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Target, 0, 0, BSCC_Reachable_States, [], Policy_BSCC, BSCC_Allowed_Actions)

                    for y in range(len(States_For_Reachability)):
                        Permanent_Policy[States_For_Reachability[y]] = Policy_BSCC[Component.index(States_For_Reachability[y])]
                    
                    List_Permanent_Acc_BSCC.append(Component)
                    
    print List_Permanent_Acc_BSCC                                     
    print List_G_Pot                    
                         


                       
    return G_Pot_Acc_BSCCs, G_Per_Acc_BSCCs, Potential_Policy, Permanent_Policy, Al_Act_Pot, Al_Act_Perm, first, Is_In_Permanent_Comp, List_Permanent_Acc_BSCC, List_G_Pot, Which_Potential_Acc_BSCC, Is_In_Potential_Acc_BSCC, Bridge_Potential_Accepting




def Maximize_Lower_Bound_Reachability(IA_l, IA_u, Q1, Num_States, Automata_size, Reach, Init, Optimal_Policy, Actions):
    
    
    Ascending_Order = []
    Index_Vector = np.zeros((IA_l.shape[1],1))
    Is_In_Q1 = np.zeros((IA_l.shape[1]))
    
    for k in range(IA_l.shape[1]):                               
        if k in Q1:            
            Index_Vector[k,0] = 1.0
            Ascending_Order.append(k)
            Is_In_Q1[k] = 1
           
        else:            
            Index_Vector[k,0] = 0.0
            Ascending_Order.insert(0,k)

    d = {k:v for v,k in enumerate(Ascending_Order)} 
    Sort_Reach = []

    for i in range(len(Reach)):
        Sort_Reach.append([])
        for j in range(IA_l.shape[1]):
            Sort_Reach[-1].append([])
    
    for j in range(IA_l.shape[1]):        
        if Is_In_Q1[j] == 0:
            for k in range(len(Actions[j])):
                Reach[Actions[j][k]][j].sort(key=d.get)
                Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
        else:
            continue
               
    Phi_Min = Phi_Synthesis_Max_Lower(IA_l, IA_u, Ascending_Order, Q1, Reach, Sort_Reach, Actions)
    Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Min), Index_Vector)
        
    for i in range(IA_l.shape[1]):
        if Is_In_Q1[i] == 1: continue
        List_Values = []
        for k in range(len(Actions[i])):
            List_Values.append(IA_l.shape[0]*i+Actions[i][k])            
        Values = Steps_Low[List_Values]
        Index_Vector[i,0] = np.amax(Values)
        Optimal_Policy[i] = Actions[i][np.argmax(Values)]
    

    for i in range(len(Q1)):    
        Index_Vector[Q1[i],0] = 1.0

    Success_Intervals = []       
    for i in range(IA_l.shape[1]):       
        Success_Intervals.append(Index_Vector[i,0]) 
        
     
    Terminate_Check = 0
    Convergence_threshold = 0.01
    Previous_Max_Difference = 1
           
    
    while Terminate_Check == 0:
                   
        Previous_List = copy.copy(Ascending_Order)
               
        for i in range(len(Q1)):
            Success_Intervals[Q1[i]] = 1.0
       
        Ascending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Ascending_Order = list(Ascending_Order[(Success_Array).argsort()]) 
        
        d = {k:v for v,k in enumerate(Ascending_Order)} 
        Sort_Reach = []

        for i in range(len(Reach)):
            Sort_Reach.append([])
            for j in range(IA_l.shape[1]):
                Sort_Reach[-1].append([])
        
        for j in range(IA_l.shape[1]):        
            if Is_In_Q1[j] == 0:
                for k in range(len(Actions[j])):
                    Reach[Actions[j][k]][j].sort(key=d.get)
                    Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
            else:
                continue
        
        if Previous_List != Ascending_Order:
            Phi_Min = Phi_Synthesis_Max_Lower(IA_l, IA_u, Ascending_Order, Q1, Reach, Sort_Reach, Actions)

        Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Min), Index_Vector)
    
        List_Values = list([])
        Bounds_All_Act = list([])
        for i in range(IA_l.shape[1]):
            if Is_In_Q1[i] == 1: 
                Bounds_All_Act.append(list([]))
                for j in range(len(Actions[i])):
                    Bounds_All_Act[-1].append(1.0)
                continue
            List_Values.append([])
            for k in range(len(Actions[i])):
                List_Values[-1].append(IA_l.shape[0]*i+Actions[i][k])
            Values = list(Steps_Low[List_Values[-1]])
            Bounds_All_Act.append(Values)
            Index_Vector[i,0] = np.amax(Values)
            Optimal_Policy[i] = Actions[i][np.argmax(Values)]    
            
        
        for i in range(len(Q1)):    
            Index_Vector[Q1[i],0] = 1.0
                         
        Max_Difference = 0
                       
        for i in range(IA_l.shape[1]):
                                  
            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Index_Vector[i,0]))        
            Success_Intervals[i] = Index_Vector[i,0]
        
            
        if Max_Difference < Convergence_threshold:              
            Terminate_Check = 1    
    
    Bounds = []
    Prod_Bounds = []
    
    Indices = [int(i*IA_l.shape[0]+Optimal_Policy[i]) for i in range(len(Optimal_Policy))]
    Phi_Min = np.array(Phi_Min[Indices,:])
    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
    for i in range(len(Success_Intervals)):
        Prod_Bounds.append(Success_Intervals[i])
        
    return (Bounds, Prod_Bounds, Phi_Min, Optimal_Policy, Bounds_All_Act)











def Maximize_Upper_Bound_Reachability(IA_l, IA_u, Q1, Num_States,Automata_size, Reach, Init, Optimal_Policy, Actions):
    
    
    Descending_Order = []
    Index_Vector = np.zeros((IA_l.shape[1],1)) 
    Is_In_Q1 = np.zeros((IA_l.shape[1]))
    
    for k in range(IA_l.shape[1]):                               
        if k in Q1:            
            Index_Vector[k,0] = 1.0
            Descending_Order.insert(0,k)
            Is_In_Q1[k] = 1
           
        else:            
            Index_Vector[k,0] = 0.0
            Descending_Order.append(k)

    d = {k:v for v,k in enumerate(Descending_Order)} 
    Sort_Reach = []

    for i in range(len(Reach)):
        Sort_Reach.append([])
        for j in range(IA_l.shape[1]):
            Sort_Reach[-1].append([])
    
    for j in range(IA_l.shape[1]):        
        if Is_In_Q1[j] == 0:
            for k in range(len(Actions[j])):
                Reach[Actions[j][k]][j].sort(key=d.get)
                Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
        else:
            continue
                
    Phi_Max = Phi_Synthesis_Max_Upper(IA_l, IA_u, Descending_Order, Q1, Reach, Sort_Reach, Actions)   
    Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Max), Index_Vector)
        
    for i in range(IA_l.shape[1]):
        if Is_In_Q1[i] == 1: continue
        List_Values = []
        for k in range(len(Actions[i])):
            List_Values.append(IA_l.shape[0]*i+Actions[i][k])
        Values = list(Steps_Low[List_Values])
        Index_Vector[i,0] = np.amax(Values)
        Optimal_Policy[i] = Actions[i][np.argmax(Values)]

    for i in range(len(Q1)):    
        Index_Vector[Q1[i],0] = 1.0

    Success_Intervals = list([])    
 
    for i in range(IA_l.shape[1]):       
        Success_Intervals.append(Index_Vector[i,0]) 


    Terminate_Check = 0
    Convergence_threshold = 0.01
    Previous_Max_Difference = 1
    count = 0
           

    while Terminate_Check == 0:
        
        count += 1
                   
        Previous_List = copy.copy(Descending_Order)
               
        for i in range(len(Q1)):
            Success_Intervals[Q1[i]] = 1.0
       
        Descending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Descending_Order = list(Descending_Order[(-Success_Array).argsort()]) 


        
        d = {k:v for v,k in enumerate(Descending_Order)} 
        Sort_Reach = list([])

        for i in range(len(Reach)):
            Sort_Reach.append([])
            for j in range(IA_l.shape[1]):
                Sort_Reach[-1].append([])
        
        for j in range(IA_l.shape[1]):        
            if Is_In_Q1[j] == 0:
                for k in range(len(Actions[j])):
                    Reach[Actions[j][k]][j].sort(key=d.get)
                    Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
            else:
                continue
        
        if Previous_List != Descending_Order:
            Phi_Max = Phi_Synthesis_Max_Upper(IA_l, IA_u, Descending_Order, Q1, Reach, Sort_Reach, Actions)

        Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Max), Index_Vector[:,0])

        List_Values = list([])
        Bounds_All_Act = list([])
    
        for i in range(IA_l.shape[1]):
            if Is_In_Q1[i] == 1:
                Bounds_All_Act.append(list([]))
                for j in range(len(Actions[i])):
                    Bounds_All_Act[-1].append(1.0)
                continue                
            List_Values.append([])         
            for k in range(len(Actions[i])):
                List_Values[-1].append(IA_l.shape[0]*i+Actions[i][k])    
            Values = list(Steps_Low[List_Values[-1]])  
            Bounds_All_Act.append(Values)
            Index_Vector[i,0] = np.amax(Values)
            Optimal_Policy[i] = Actions[i][np.argmax(Values)] 
        
        for i in range(len(Q1)):    
            Index_Vector[Q1[i],0] = 1.0
                                   
        Max_Difference = 0
        
               
        for i in range(IA_l.shape[1]):                                
            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Index_Vector[i,0]))        
            Success_Intervals[i] = Index_Vector[i,0]
         
        if Max_Difference < Convergence_threshold:              
            Terminate_Check = 1    
    
    Bounds = []
    Prod_Bounds = []
    
    Indices = [int(i*IA_l.shape[0]+Optimal_Policy[i]) for i in range(len(Optimal_Policy))]
    Phi_Max = np.array(Phi_Max[Indices,:])
    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
    for i in range(len(Success_Intervals)):
        Prod_Bounds.append(Success_Intervals[i])
        
    return (Bounds, Prod_Bounds, Phi_Max, Optimal_Policy, Bounds_All_Act)




def Phi_Synthesis_Max_Lower(Lower, Upper, Order_A, q1, Reach, Reach_Sort, Action):
    
    Phi_min = np.zeros((Upper.shape[1]*Upper.shape[0], Upper.shape[1]))

    for j in range(Upper.shape[1]):
        
        if j in q1:
            continue
        else:
    
            for k in range(len(Action[j])):
                Up = Upper[Action[j][k]][j][:]
                Low = Lower[Action[j][k]][j][:]                 
                Sum_1_A = 0.0
                Sum_2_A = sum(Low[Reach[Action[j][k]][j]])    
                Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][0]] = min(Low[Reach_Sort[Action[j][k]][j][0]] + 1 - Sum_2_A, Up[Reach_Sort[Action[j][k]][j][0]])  
          
                for i in range(1, len(Reach_Sort[Action[j][k]][j])):
                                 
                    Sum_1_A = Sum_1_A + Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i-1]]
                    if Sum_1_A >= 1:
                        break
                    Sum_2_A = Sum_2_A - Low[Reach_Sort[Action[j][k]][j][i-1]]
                    Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i]] = min(Low[Reach_Sort[Action[j][k]][j][i]] + 1 - (Sum_1_A+Sum_2_A), Up[Reach_Sort[Action[j][k]][j][i]])                 
    return Phi_min





def Phi_Synthesis_Max_Upper(Lower, Upper, Order_D, q1, Reach, Reach_Sort, Action):
    
    Phi_max = np.zeros((Upper.shape[1]*Upper.shape[0], Upper.shape[1]))
    
    for j in range(Upper.shape[1]):
        
        if j in q1:
            continue
        else:
    
            for k in range(len(Action[j])):

                Up = Upper[Action[j][k]][j][:]
                Low = Lower[Action[j][k]][j][:] 
                Sum_1_D = 0.0
                Sum_2_D = sum(Low[Reach[Action[j][k]][j]])
                Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][0]] = min(Low[Reach_Sort[Action[j][k]][j][0]] + 1 - Sum_2_D, Up[Reach_Sort[Action[j][k]][j][0]])  
          
                for i in range(1, len(Reach_Sort[Action[j][k]][j])):
                                 
                    Sum_1_D = Sum_1_D + Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i-1]]
                    if Sum_1_D >= 1:
                        break
                    Sum_2_D = Sum_2_D - Low[Reach_Sort[Action[j][k]][j][i-1]]
                    Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i]] = min(Low[Reach_Sort[Action[j][k]][j][i]] + 1 - (Sum_1_D+Sum_2_D), Up[Reach_Sort[Action[j][k]][j][i]])  
               
    return Phi_max


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


