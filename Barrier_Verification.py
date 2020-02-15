#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:05:11 2019

@author: maxencedutreix
"""

import numpy as np
import Barrier_Verification_Functions as func
import matplotlib.pyplot as plt
import timeit
import sys
import pickle

sys.setrecursionlimit(10000)

#np.set_printoptions(threshold='NaN')
start = timeit.default_timer()
plt.close("all")
Space_Tag = 0
I_d = 0.01 #Refinement Threshold
V_Stop = 0.04
First_Verif = 1


State_Space = np.array( [
                [[-0.5,-0.5],[-0.25,-0.25]],
                [[-0.25,-0.5],[0.0,-0.25]],
                [[0.0,-0.5],[0.25,-0.25]],
                [[0.25,-0.5],[0.5,-0.25]], 
                [[-0.5,-0.25],[-0.25,0.0]],
                [[-0.25,-0.25],[0.0,-0.125]],
                [[-0.25,-0.125],[0.0,0.0]],
                [[0.0,-0.25],[0.25, -0.125]],
                [[0.0,-0.125],[0.25, 0.0]],
                [[0.25, -0.25],[0.5, 0.0]], 
                [[-0.5, 0.],[-0.25, 0.25]],
                [[-0.25,0.0],[0.0,0.125]],
                [[-0.25,0.125],[0.0,0.25]],
                [[0.0,0.0],[0.25,0.125]],
                [[0.0,0.125],[0.25,0.25]],
                [[0.25,0.0],[0.5,0.25]], 
                [[-0.5,0.25],[-0.25,0.5]],
                [[-0.25,0.25],[0.0,0.5]],
                [[0.0,0.25],[0.25,0.5]],
                [[0.25,0.25],[0.5,0.5]],                 
               ] )

L_mapping = ['', '', '', 'B',
             '', '', '', '' ,'', 'C',
             'C', '', '', '' , '', '',
             'B', 'A', '', '']

for i in range(3):
    
    Set_Refinement = range(State_Space.shape[0])
    New_States = func.Raw_Refinement(Set_Refinement, State_Space)
    
    for m in range(len(Set_Refinement)):
                                   
            State_Space = np.insert(State_Space, Set_Refinement[m]+1+m , np.asarray(New_States[2*m]), 0)
            State_Space = np.insert(State_Space, Set_Refinement[m]+1+m , np.asarray(New_States[2*m+1]), 0)           
            State_Space = np.delete(State_Space, Set_Refinement[m]+m, 0) 
    
    for m in range(len(Set_Refinement)):
       L_mapping.insert(Set_Refinement[m]+m+1, L_mapping[Set_Refinement[m]+m])



Domain = [-0.5, -0.5, 0.5, 0.5] #Coordinates of domain


#Pre_processing step
Reachable_States = [[] for i in range(State_Space.shape[0])]

for i in range(State_Space.shape[0]):
    Reachable_x1_set = [6.0*((State_Space[i][0][0])**3)*State_Space[i][0][1], 6.0*((State_Space[i][1][0])**3)*State_Space[i][1][1]]
    if Reachable_x1_set[0] > Reachable_x1_set[1]:
        Reachable_x1_set = list([Reachable_x1_set[1], Reachable_x1_set[0]])
        
    for j in range(State_Space.shape[0]):
        if Reachable_x1_set[0] > State_Space[j][1][0] or Reachable_x1_set[1] < State_Space[j][0][0]:
            continue
        Reachable_States[i].append(j)

start_index = [0,0]


(Lower_Bound_Matrix, Upper_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States, Known_Bounds) = func.Probability_Interval_Computation_Barrier(State_Space, Domain, Reachable_States, start_index) 


#(Lower_Bound_Matrix, Upper_Bound_Matrix) = func.Bounds_Tightening(Lower_Bound_Matrix, Upper_Bound_Matrix)







Automata = [[[], ['', 'A'],[], ['B'], ['C'], []], 
             [[], [],[''], ['B'], ['C', 'A'], []],
             [[], [], [], ['B'],['A','C'], ['']],
             [[], [] ,[] , ['', 'B', 'A', 'C'], [], []],
             [[], [] ,[] , ['B'], ['', 'A', 'C'],[]], 
             [[], [] ,[] , ['B'], ['C'],['', 'A']]]




#Automata_Accepting contains the Rabin Pairs (Ei, Fi) of Automata, with Fi being the 'good' states
Automata_Accepting = [[[],[4]]]
p_threshold = 0.82
Order = 2


#Constructs the product between the IMC and the Automata
(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Init) = func.Build_Product_IMC(Lower_Bound_Matrix, Upper_Bound_Matrix, Automata, L_mapping, Automata_Accepting, Reachable_States, Is_Bridge_State, Bridge_Transitions)

#Contains the interval of satisfactions for all states
Success_Intervals = [[] for n in range(State_Space.shape[0])]

#Contains the probability of reaching an accepting BSCC from all states in the product IMC
Product_Intervals = [[] for n in range(IA1_l.shape[0])]




List_Inevitable_A_BSCC = []
List_Inevitable_Non_A_BSCC = []
Is_in_P = np.zeros(IA1_l.shape[0])
Leaky_States_L_Accepting = [] #Contains the leaky states with respect to the Largest A BSCC
Leaky_States_L_Non_Accepting = [] #Contains the leaky states with respect to the Largest Non A BSCC
Leaky_States_P_Accepting = [] #Contains the leaky states with respect to the permanent A BSCC
Leaky_States_P_Non_Accepting = [] #Contains the leaky states with respect to the permanent Non A BSCC
Leaky_L_Accepting_Previous = [] #Contains leaky states of Largest A BSCCs at previous step
Leaky_L_Non_Accepting_Previous = [] #Contains leaky states of Largest non A BSCCs at previous step
Previous_A_BSCC = [] #Contains Largest A BSCCs at previous step
Previous_Non_A_BSCC = [] #Contains Largest Non A BSCCs at previous step


(List_Largest_Non_A_BSCC, List_Largest_A_BSCC, List_Inevitable_Non_A_BSCC, List_Inevitable_A_BSCC, Is_State_In_L_A, Is_State_In_L_N_A, Which_Acc_BSCC, Which_N_Acc_BSCC, Bridge_Acc, Bridge_N_Acc, Graph, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting) = func.Find_Largest_BSCCs_One_Pair(IA1_l, IA1_u, Automata_Accepting, Lower_Bound_Matrix.shape[0], Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Is_in_P, List_Inevitable_A_BSCC, List_Inevitable_Non_A_BSCC, Previous_A_BSCC, Previous_Non_A_BSCC, First_Verif)

(List_Largest_Winning_Components, Inevitable_Winning_Components, List_Largest_Losing_Components, Inevitable_Losing_Components, Largest_Winning_Components, Largest_Losing_Components, Is_in_WC_L, Which_WC_L, Bridge_WC_L, Is_in_LC_L, Which_LC_L, Bridge_LC_L, Is_in_WC_P, Which_WC_P, Bridge_WC_P, Is_in_LC_P, Which_LC_P, Bridge_LC_P, Is_in_P, List_Inevitable_A_BSCC, List_Inevitable_Non_A_BSCC, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC) = func.Find_Winning_Losing_Components(IA1_l, IA1_u, List_Largest_Non_A_BSCC, List_Largest_A_BSCC, List_Inevitable_Non_A_BSCC, List_Inevitable_A_BSCC, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Graph, Bridge_Acc, Bridge_N_Acc, Is_State_In_L_A, Is_State_In_L_N_A, Is_in_P, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC)



if len(Largest_Losing_Components) == 0: 
    for i in range(len(Success_Intervals)):
        Success_Intervals[i].append(1.0)
        Success_Intervals[i].append(1.0)
        
           
        
elif len(Largest_Winning_Components) == 0:

    for i in range(len(Success_Intervals)):
        Success_Intervals[i].append(0.0)
        Success_Intervals[i].append(0.0)
        

else:
    
    if len(Inevitable_Winning_Components) ==  0:
        Low_Bounds = [0.0 for j in range(State_Space.shape[0])]
        Low_Bounds_Prod = [0.0 for j in range(IA1_u.shape[0]) ]
        Worst_Markov_Chain = func.Random_Markov_Chain_IMC(IA1_u, IA1_l)
    else:    
        (Prob_Reach, Low_Bounds_Prod, Worst_Markov_Chain) = func.Reachability_Upper(IA1_l, IA1_u, Largest_Losing_Components, Inevitable_Winning_Components, Lower_Bound_Matrix.shape[0], len(Automata), Product_Reachable_States, Init)
        Low_Bounds = [i - j for i, j in zip([1.0]*len(Prob_Reach), Prob_Reach)]
        Low_Bounds_Prod = [i - j for i, j in zip([1.0]*len(Low_Bounds_Prod), Low_Bounds_Prod)]
            
    if len(Inevitable_Losing_Components) == 0:
        Upp_Bounds = [1.0 for j in range(State_Space.shape[0])]
        Upp_Bounds_Prod = [1.0 for j in range(IA1_u.shape[0]) ]
        Best_Markov_Chain = func.Random_Markov_Chain_IMC(IA1_u, IA1_l)                        
    else:
        (Upp_Bounds, Upp_Bounds_Prod, Best_Markov_Chain) = func.Reachability_Upper(IA1_l, IA1_u, Largest_Winning_Components, Inevitable_Losing_Components, Lower_Bound_Matrix.shape[0], len(Automata), Product_Reachable_States, Init)

    for i in range(len(Success_Intervals)):   
        Success_Intervals[i].append(Low_Bounds[i])
        Success_Intervals[i].append(Upp_Bounds[i])
    
    for i in range(len(Product_Intervals)):
        Product_Intervals[i].append(Low_Bounds_Prod[i])
        Product_Intervals[i].append(Upp_Bounds_Prod[i])
            
        


Yes_States = []
No_States = []
Maybe_States = []
Max_Int = 0.0

if Order == 1:
    ### Probability <=
    
    for i in range(len(Success_Intervals)): 
        if Success_Intervals[i][1] <= p_threshold:     
            Yes_States.append(i)
            
        elif Success_Intervals[i][0] > p_threshold:       
            No_States.append(i)
            
        else:
            Max_Int = max(Max_Int, Success_Intervals[i][1] - Success_Intervals[i][0])
            Maybe_States.append(i)
    
elif Order == 2: 
    ### Probability >=
    
    for i in range(len(Success_Intervals)):  
        if Success_Intervals[i][1] < p_threshold:     
            No_States.append(i)
            
        elif Success_Intervals[i][0] >= p_threshold:       
            Yes_States.append(i)
            
        else:
            Max_Int = max(Max_Int, Success_Intervals[i][1] - Success_Intervals[i][0])
            Maybe_States.append(i)
    
elif Order == 3:   
    ### Probability <

    for i in range(len(Success_Intervals)):       
        if Success_Intervals[i][1] < p_threshold:     
            Yes_States.append(i)
            
        elif Success_Intervals[i][0] >= p_threshold:       
            No_States.append(i)
            
        else:
            
            Max_Int = max(Max_Int, Success_Intervals[i][1] - Success_Intervals[i][0])
            Maybe_States.append(i)
else:      
    ### Probability >

    for i in range(len(Success_Intervals)):     
        if Success_Intervals[i][1] <= p_threshold:     
            No_States.append(i)
            
        elif Success_Intervals[i][0] > p_threshold:       
            Yes_States.append(i)
            
        else:
            Max_Int = max(Max_Int, Success_Intervals[i][1] - Success_Intervals[i][0])
            Maybe_States.append(i)    


Space_Tag = func.State_Space_Plot(State_Space, Yes_States, No_States, Maybe_States, Space_Tag)


print 'TOTAL RUN TIME'
print timeit.default_timer() - start


