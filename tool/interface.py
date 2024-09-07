'''
    operation - Gyuyeon Lim (lky473736)
    tool/interface (main) 
'''

import pandas as pd
import numpy as np
import os
import math
import operator
import make_data

operators = operator.Operator.operators

# interface
def interface (section, num_operand=2, num_record=2, start_num=1, 
               path='../data/', operator_input=None) :
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
        
    if path.find('../data/') != True : 
        print ("ERROR!")
        print ("The path is incorrect : main directory is wrong.")
        token = 1
        
    if section not in (1, 2, 3) :
        print ("ERROR!")
        print ("Undefined section.")
        
    if token == 1 : 
        exit()
        
    print ("operators that can calculate : ", operators)
    print ("operator that chosen : ", operator_input)
    print ("section : ", section)
    print ("")
    
    # if operator is real thing?
    if operator in operators.keys() : 
        # 허용된 함수 및 연산자만 사용할 수 있도록 제한
        
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
   
    else : 
        print ("ERROR!")
        print ("This operator cannot be used.")
        exit()
