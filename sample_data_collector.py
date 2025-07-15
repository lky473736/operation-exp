#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath('.'))

import tool.interface as interface

interface.interface(
    section=1,
    num_operand=2,
    num_record=100,
    path='./data/section1/section1_add.csv',
    operator_input='+'      
)

interface.interface(
    section=2,
    num_operand=3,
    num_record=100,         
    path='./data/section2/section2_mixed.csv'
)

interface.interface(
    section=3,
    num_operand=3,
    num_record=100,
    path='./data/section3/section3_formula.csv',
    custom_operators=['+', '*']  # operand1 + operand2 * operand3
)

interface.interface(
    section=3,
    num_operand=2,
    num_record=100,
    path='./data/section3/section3_**.csv',
    custom_operators=['**'] # operand1 ** operand2
)

interface.interface(
    section=3,
    num_operand=4,
    num_record=100,
    path='./data/section3/section3_max_-_*.csv',
    custom_operators=['max', '-', '*'] # max(op1,op2) - op3 * op4
)