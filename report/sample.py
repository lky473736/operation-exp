import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

print("Current working directory:", os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
print("Current sys.path:", sys.path)

import tool.interface as interface
from tool.operator import Operator
operators = Operator.operators

for operator in operators.keys() : # basically testing two operands using all operators
    interface.interface(section=1, num_operand=2, num_record=100,
               path=parent_dir+f'/operation/data/sample/sample_{operators[operator][0]}.csv', 
               operator_input=operator)