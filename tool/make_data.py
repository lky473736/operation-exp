'''
    operation - Gyuyeon Lim (lky473736)
    tool/make_data
'''

import pandas as pd
import numpy as np
import os
import math
import operator

from tool.operator import Operator
operators = Operator.operators

def make_data_section1 (num_record, num_operand, operator_input) :
    columns = [f'operand{i}' for i in range (1, num_operand+1)]
    columns.append ('target')
    
    df = pd.DataFrame(columns=columns)
        
    for i in range (1, num_record+1) :
        record = operators[operator_input](num_operand, i)
            
        if record == None : 
            print ("ERROR!")
            print ("This operator cannot calculate more than two operand.")
            exit()
                
        else : 
            # df = df.append(record, ignore_index=True)
            record_df = pd.DataFrame([record])
            df = pd.concat([df, record_df], ignore_index=True)
            
    return df

# def make_data_section2 (num_record, num_operand, operator_input) :
#     columns = [f'operand{i}' for i in range (1, num_operand+1)]
#     columns.append ('result')
#     columns.append ('target') # operator
    
#     df = pd.DataFrame(columns=columns)
        
#     for i in range (num_record) :
#         record = operator[operator_input](num_operand, i)
            
#         if record == None : 
#             print ("ERROR!")
#             print ("This operator cannot calculate more than two operand.")
#             exit()
                
#         else : 
#             df = df.append(record, ignore_index=True)
            
#     return df