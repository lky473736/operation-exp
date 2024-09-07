import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

print("Current working directory:", os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
print("Current sys.path:", sys.path)
import tool.interface as interface

interface.interface(section=1, num_operand=2, num_record=100,
               path=parent_dir+'/operation/data/sample.csv', operator_input='+')