'''
    operation - Gyuyeon Lim (lky473736)
    tool/make_data
'''

import pandas as pd
import numpy as np
import os
import math
import operator
import random

from tool.operator import Operator
operators = Operator.operators

def make_data_section1 (num_record, num_operand, operator_input) :
    # Create columns for operands and target
    columns = [f'operand{i}' for i in range (1, num_operand+1)]
    columns.append ('target')
    
    df = pd.DataFrame(columns=columns)
        
    for i in range (1, num_record+1) :
        record = operators[operator_input][1](num_operand)
            
        if record == None : 
            print ("ERROR!")
            print ("This operator cannot calculate more than two operand.")
            exit()
                
        else : 
            record_df = pd.DataFrame([record])
            df = pd.concat([df, record_df], ignore_index=True)
            
    return df

def make_data_section2(num_record, num_operand):
    # Mix all operators for classification task
    columns = [f'operand{i}' for i in range(1, num_operand+1)]
    columns.append('result')
    columns.append('target')  # operator name
    
    df = pd.DataFrame(columns=columns)
    
    # Generate data for each operator
    for operator_symbol in operators.keys():
        if operators[operator_symbol][1](num_operand) is not None:  # check if operator supports this num_operand
            for i in range(num_record):
                record_data = operators[operator_symbol][1](num_operand)
                
                if record_data is not None:
                    # Convert for Section 2: target->result, operator->target
                    new_record = {}
                    for j in range(1, num_operand+1):
                        new_record[f'operand{j}'] = record_data[f'operand{j}']
                    new_record['result'] = record_data['target']
                    new_record['target'] = operators[operator_symbol][0]  # operator name
                    
                    record_df = pd.DataFrame([new_record])
                    df = pd.concat([df, record_df], ignore_index=True)
    
    return df

def make_data_section3(num_record, num_operand, custom_operators):
    # Generate data with custom formula - target is result of specific formula
    columns = [f'operand{i}' for i in range(1, num_operand+1)]
    columns.append('target')
    
    df = pd.DataFrame(columns=columns)
    
    for i in range(num_record):
        # Generate random operands
        components = [random.randint(0, 101) for j in range(num_operand)]
        
        record = {}
        for j in range(num_operand):
            record[f'operand{j+1}'] = components[j]
        
        # Apply custom_operators in sequence
        result = components[0]
        
        for j, op in enumerate(custom_operators):
            if j + 1 < len(components) and op in operators:
                # Handle binary operators only
                if operators[op][1](2) is not None:  # check if supports 2 operands
                    try:
                        if op == '+':
                            result = result + components[j + 1]
                        elif op == '-':
                            result = result - components[j + 1]
                        elif op == '*':
                            result = result * components[j + 1]
                        elif op == '/':
                            if components[j + 1] != 0:
                                result = result / components[j + 1]
                            else:
                                result = None
                        elif op == '%':
                            if components[j + 1] != 0:
                                result = result % components[j + 1]
                            else:
                                result = None
                        elif op == '**':
                            result = result ** components[j + 1]
                        elif op == '//':
                            if components[j + 1] != 0:
                                result = result // components[j + 1]
                            else:
                                result = None
                        elif op == '&':
                            result = int(result) & components[j + 1]
                        elif op == '|':
                            result = int(result) | components[j + 1]
                        elif op == '^':
                            result = int(result) ^ components[j + 1]
                        elif op == '<<':
                            result = int(result) << components[j + 1]
                        elif op == '>>':
                            result = int(result) >> components[j + 1]
                        elif op == 'max':
                            result = max(result, components[j + 1])
                        elif op == 'min':
                            result = min(result, components[j + 1])
                        elif op == 'hypot':
                            result = math.hypot(result, components[j + 1])
                        elif op == 'gcd':
                            if result != 0 and components[j + 1] != 0:
                                result = Operator.euclidean(int(result), components[j + 1])
                            else:
                                result = None
                        elif op == 'abssub':
                            result = abs(result - components[j + 1])
                        elif op == 'log':
                            if result > 0 and components[j + 1] > 0 and components[j + 1] != 1:
                                result = math.log(result, components[j + 1])
                            else:
                                result = None
                    except (ValueError, ZeroDivisionError, OverflowError):
                        result = None
                        break
        
        record['target'] = result
        
        # Only add records with valid results
        if result is not None:
            record_df = pd.DataFrame([record])
            df = pd.concat([df, record_df], ignore_index=True)
    
    return df