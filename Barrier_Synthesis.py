#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 03:18:34 2019

@author: maxencedutreix
"""

import numpy as np
import Barrier_Synthesis_Functions as func
import matplotlib.pyplot as plt
import timeit
import sys
import copy
import pickle

sys.setrecursionlimit(10000)
start = timeit.default_timer()
plt.close("all")
Space_Tag = 0


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



Automata = [[[], ['', 'A'],[], ['B'], ['C'], []], 
             [[], [],[''], ['B'], ['C', 'A'], []],
             [[], [], [], ['B'],['A','C'], ['']],
             [[], [] ,[] , ['', 'B', 'A', 'C'], [], []],
             [[], [] ,[] , ['B'], ['', 'A', 'C'],[]], 
             [[], [] ,[] , ['B'], ['C'],['', 'A']]]

Automata_Accepting = [[[],[3]]]

Reachable_States = [[[] for x in range(State_Space.shape[0])] for y in range(2)]


for i in range(State_Space.shape[0]):
    Reachable_x1_set = [6.0*((State_Space[i][0][0])**3)*State_Space[i][0][1], 6.0*((State_Space[i][1][0])**3)*State_Space[i][1][1]]
    if Reachable_x1_set[0] > Reachable_x1_set[1]:
        Reachable_x1_set = list([Reachable_x1_set[1], Reachable_x1_set[0]])
        
    for j in range(State_Space.shape[0]):
        if Reachable_x1_set[0] > State_Space[j][1][0] or Reachable_x1_set[1] < State_Space[j][0][0]:
            continue
        Reachable_States[0][i].append(j)

    Reachable_x1_set = [7.0*((State_Space[i][0][0])**3)*State_Space[i][0][1], 7.0*((State_Space[i][1][0])**3)*State_Space[i][1][1]]
    if Reachable_x1_set[0] > Reachable_x1_set[1]:
        Reachable_x1_set = list([Reachable_x1_set[1], Reachable_x1_set[0]])
        
    for j in range(State_Space.shape[0]):
        if Reachable_x1_set[0] > State_Space[j][1][0] or Reachable_x1_set[1] < State_Space[j][0][0]:
            continue
        Reachable_States[1][i].append(j)        




(Lower_Bound_Matrix, Upper_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States) = func.BMDP_Probability_Interval_Computation_Barrier(State_Space, Domain, Reachable_States) 
(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Init) = func.Build_Product_BMDP(Lower_Bound_Matrix, Upper_Bound_Matrix, Automata, L_mapping, Automata_Accepting, Reachable_States, Is_Bridge_State, Bridge_Transitions)




Allowable_Actions = []
for i in range(len(State_Space)*len(Automata)):
    Allowable_Actions.append(range(2))

first = 1;
Allowable_Action_Potential = list([]) 
Allowable_Action_Permanent = list([]) 
Is_In_Permanent_Comp = np.zeros(IA1_l.shape[1])  
Is_In_Permanent_Comp = Is_In_Permanent_Comp.astype(int)
List_Permanent_Accepting_BSCC = [] 
List_Permanent_Non_Accepting_BSCC = list([])
Previous_Accepting_BSCC = list([])
Previous_Non_Accepting_BSCC = list([])

Optimal_Policy = np.zeros(IA1_l.shape[1])
Optimal_Policy = Optimal_Policy.astype(int)
Potential_Policy = np.zeros(IA1_l.shape[1]) #Policy to generate the "best" best-case (maximize upper bound)
Potential_Policy = Potential_Policy.astype(int)
   
(Greatest_Potential_Accepting_BSCCs, Greatest_Permanent_Accepting_BSCCs, Potential_Policy, Optimal_Policy, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, List_Potential_Accepting_BSCC, Which_Potential_Accepting_BSCC, Is_In_Potential_Accepting_BSCC, Bridge_Accepting_BSCC) = func.Find_Greatest_Accepting_BSCCs(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States, Product_Bridge_Transitions, Product_Is_Bridge_State, Automata_Accepting, Potential_Policy, Optimal_Policy, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, Previous_Accepting_BSCC) # Will return greatest potential accepting bsccs


Previous_Accepting_BSCC = list(List_Potential_Accepting_BSCC)

Allowable_Action_Permanent = copy.deepcopy(Allowable_Action_Potential) #Some potential actions could be permanent actions for certain states under refinement 
      
for i in range(len(Greatest_Permanent_Accepting_BSCCs)):
    Allowable_Actions[Greatest_Permanent_Accepting_BSCCs[i]] = list([Optimal_Policy[Greatest_Permanent_Accepting_BSCCs[i]]])

    
   
Which_Potential_Accepting_BSCC = Which_Potential_Accepting_BSCC.astype(int)
Greatest_Permanent_Winning_Component = list(Greatest_Permanent_Accepting_BSCCs)
Greatest_Potential_Winning_Component = list(Greatest_Potential_Accepting_BSCCs)
Is_In_Potential_Winning_Component = list(Is_In_Potential_Accepting_BSCC)
List_Potential_Winning_Components = copy.deepcopy(List_Potential_Accepting_BSCC)
Bridge_Winning_Components = copy.deepcopy(Bridge_Accepting_BSCC)
Bridge_Winning_Components = [[Bridge_Winning_Components[i]] for i in range(len(Bridge_Winning_Components))]
Which_Potential_Winning_Component = [[] for i in range(IA1_l.shape[1])]   
for i in range(len(Which_Potential_Accepting_BSCC)):
   if Is_In_Potential_Winning_Component[i] == 1:
       Which_Potential_Winning_Component[i].append([Which_Potential_Accepting_BSCC[i],0]) 
       
    
Reach_Allowed_Actions = []
for y in range(IA1_l.shape[1]): # For all the system states
    Reach_Allowed_Actions.append(Allowable_Actions[y])

    
(Low_Bound, Low_Bounds_Prod, Worst_Markov_Chain, Optimal_Policy, List_Values_Low) = func.Maximize_Lower_Bound_Reachability(IA1_l, IA1_u, Greatest_Permanent_Winning_Component, State_Space.shape[0], len(Automata), Product_Reachable_States, Init, Optimal_Policy, Reach_Allowed_Actions) # Maximizes Lower Bound
(Upp_Bound, Upp_Bounds_Prod, Best_Markov_Chain, Potential_Policy, List_Values_Up) = func.Maximize_Upper_Bound_Reachability(IA1_l, IA1_u, Greatest_Potential_Winning_Component, State_Space.shape[0], len(Automata), Product_Reachable_States, Init, Potential_Policy, Reach_Allowed_Actions) # Maximizes for winning component

Prob_Success = func.Simulation(State_Space, L_mapping, Automata, Optimal_Policy, Init)
