'''
    operation - Gyuyeon Lim (lky473736)
    tool/interface (main) 
'''

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import numpy as np
import os
import math
import tool.operator as operator
import tool.make_data as make_data

from tool.operator import Operator
operators = Operator.operators

# interface
def interface (section, num_operand=2, num_record=2, 
               path='./', operator_input=None) :
    token = 0
    
    if num_operand <= 1 : # unary?
        print ("ERROR!")
        print ("Do not make the operation by unary calculating or non-planning calculating.")
        token = 1
        
    if num_record < 2 : # just one record?
        print ("ERROR!")
        print ("It cannot make just one record for fitting the data.")
        token = 1
            
    if os.path.isfile(path) : # already has?
        print ("ERROR!")
        print ("The file is already at the path you input.")
        token = 1
        
    # if path.find('../data/') != True : 
    #     print ("ERROR!")
    #     print ("The path is incorrect : main directory is wrong.")
    #     token = 1
        
    if section not in (1, 2, 3) :
        print ("ERROR!")
        print ("Undefined section.")
        
    if token == 1 : 
        exit()
        
    print ("operators that can calculate : ", operators.keys())
    print ("operator that chosen : ", operator_input)
    print ("section : ", section)
    
    # if operator is real thing?
    if operator_input in operators.keys() :         
        df = pd.DataFrame()
        
        match (section) :
            case 1 : 
                df = make_data.make_data_section1(num_record, 
                                                  num_operand,
                                                  operator_input)
                
            case 2 : 
                # df = make_data.make_data_section2()
                pass
                
            case 3 : 
                # df = make_data_custom()    
                pass
            
        df.to_csv (path_or_buf=path, sep=',', na_rep='NaN') 
        
    else : 
        print ("ERROR!")
        print ("This operator cannot be used.")
        exit()
