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

def interface (section, num_operand=2, num_record=2, 
               path='./', operator_input=None, custom_operators=None) :
    token = 0
    
    if num_operand <= 1 : # unary?
        print ("ERROR!")
        print ("Do not make the operation by unary calculating or non-planning calculating.")
        token = 1
        
    if num_record < 2 : # just one record?
        print ("ERROR!")
        print ("It cannot make just one record for fitting the data.")
        token = 1
    
    if section not in (1, 2, 3) :
        print ("ERROR!")
        print ("Undefined section.")
        token = 1
        
    # Auto generate filename for section 1 and 3
    if section == 1 and operator_input is not None:
        if os.path.isdir(path):  # if path is directory
            filename = f"section1_{operators[operator_input][0]}.csv"
            path = os.path.join(path, filename)
    elif section == 3 and custom_operators is not None:
        if os.path.isdir(path):  # if path is directory
            filename = f"section3_{'_'.join(custom_operators)}.csv"
            path = os.path.join(path, filename)
    
    if os.path.isfile(path) : # already has?
        print ("ERROR!")
        print ("The file is already at the path you input.")
        token = 1
        
    if token == 1 : 
        exit()
        
    print ("operators that can calculate : ", operators.keys())
    print ("section : ", section)
    
    if section == 1:
        if operator_input not in operators.keys():
            print ("ERROR!")
            print ("This operator cannot be used.")
            exit()
        print ("operator that chosen : ", operator_input)
    
    df = pd.DataFrame()
    
    match (section) :
        case 1 : 
            df = make_data.make_data_section1(num_record, 
                                              num_operand,
                                              operator_input)
            
        case 2 : 
            df = make_data.make_data_section2(num_record, num_operand)
            
        case 3 : 
            df = make_data.make_data_section3(num_record, num_operand, custom_operators)
        
    df.to_csv (path_or_buf=path, sep=',', na_rep='NaN') 
    print(f"Data saved to: {path}")