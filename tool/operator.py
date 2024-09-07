'''
    operation - Gyuyeon Lim (lky473736)
    tool/operator
'''

import math
import numpy as np
import pandas as pd

class Operator (object) :
    operators = {
        '+': ['add', lambda num_operand, i: Operator.add(num_operand, i)],
        '-': ['sub', lambda num_operand, i: Operator.sub(num_operand, i)],
        '*': ['mul', lambda num_operand, i: Operator.mul(num_operand, i)],
        '/': ['truediv', lambda num_operand, i: Operator.truediv(num_operand, i)],
        '%': ['mod', lambda num_operand, i: Operator.mod(num_operand, i)],
        '**': ['pow', lambda num_operand, i: Operator.pow(num_operand, i)],
        '//': ['floordiv', lambda num_operand, i: Operator.floordiv(num_operand, i)],

        '==': ['eq', lambda num_operand, i: Operator.eq(num_operand, i)],
        '!=': ['ne', lambda num_operand, i: Operator.ne(num_operand, i)],
        '>': ['gt', lambda num_operand, i: Operator.gt(num_operand, i)],
        '<': ['lt', lambda num_operand, i: Operator.lt(num_operand, i)],
        '>=': ['ge', lambda num_operand, i: Operator.ge(num_operand, i)],
        '<=': ['le', lambda num_operand, i: Operator.le(num_operand, i)],

        '&': ['bit_and', lambda num_operand, i: Operator.bit_and(num_operand, i)],
        '|': ['bit_or', lambda num_operand, i: Operator.bit_or(num_operand, i)],
        '^': ['bit_xor', lambda num_operand, i: Operator.bit_xor(num_operand, i)],
        '<<': ['lshift', lambda num_operand, i: Operator.lshift(num_operand, i)],
        '>>': ['rshift', lambda num_operand, i: Operator.rshift(num_operand, i)],

        'divmod': ['divmod', lambda num_operand, i: Operator.divmod(num_operand, i)],
        'max': ['max', lambda num_operand, i: Operator.max(num_operand, i)],
        'min': ['min', lambda num_operand, i: Operator.min(num_operand, i)],
        'hypot': ['hypot', lambda num_operand, i: Operator.hypot(num_operand, i)],

        'gcd': ['gcd', lambda num_operand, i: Operator.gcd(num_operand, i)],
        'lcm': ['lcm', lambda num_operand, i: Operator.lcm(num_operand, i)],
        'log': ['log', lambda num_operand, i: Operator.log(num_operand, i)],
        'abssub': ['abssub', lambda num_operand, i: Operator.abssub(num_operand, i)]
    }
    
    def montage_record (num_operand) :
        feature, target = [], None
        for i in range (1, num_operand+1) :
            feature.append(f'operand{i}')
        
        return feature, target
    
    def euclidean (X, Y) :
        a = Y 
        b = X % Y 
        p = 0
        
        if b == 0 :
            return a
        
        while True :
            p = a % b
            
            if p == 0 :
                break
            
            a = b
            b = p
            
        return b
    
    @staticmethod
    def add(num_operand, i) :
        feature, target = Operator.montage_record(num_operand)
        components = [num for num in range (i, i+num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]
            
        record['target'] = sum(components)
        
        return record
    
    @staticmethod
    def sub(num_operand, i) :
        feature, target = Operator.montage_record(num_operand)
        components = [num for num in range (i, i+num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]
        
        for i in range (num_operand - 1)  :
            target = components[i] - components[i+1]

        record['target'] = target
        
        return record
    
    @staticmethod
    def mul(num_operand, i) :
        feature, target = Operator.montage_record(num_operand)
        components = [num for num in range (i, i+num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]
        
        target = 1
        for i in range (num_operand)  :
            target = target * components[i]

        record['target'] = target
        
        return record
    
    @staticmethod
    def truediv(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] / components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def floordiv(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] // components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def mod(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] % components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def pow(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] ** components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def bit_and(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] & components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def bit_or(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] | components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def bit_xor(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] ^ components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def lshift(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] << components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def rshift(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] >> components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def divmod(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = sum(divmod(components[0], components[1])) # divmod's target is the sumation share and remainder

            return record
        
        else : 
            return None
    
    @staticmethod
    def max(num_operand, i) :
        feature, target = Operator.montage_record(num_operand)
        components = [num for num in range (i, i+num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]

        record['target'] = max(components)
        
        return record
    
    @staticmethod
    def min(num_operand, i) :
        feature, target = Operator.montage_record(num_operand)
        components = [num for num in range (i, i+num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]

        record['target'] = min(components)
        
        return record
    
    @staticmethod
    def hypot(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = math.hypot(components[0], 
                                          components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def gcd(num_operand, i) :
        feature, target = Operator.montage_record(num_operand)
        components = [num for num in range (i, i+num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]

        gcd = components[0]
        for i in range (num_operand) : 
            gcd = Operator.euclidean(gcd, components[i])
        
        record['target'] = gcd
                    
        return record
    
    @staticmethod
    def lcm(num_operand, i) :
        feature, target = Operator.montage_record(num_operand)
        components = [num for num in range (i, i+num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]

        lcm = components[0]
        for i in range (num_operand) : 
            lcm = Operator.euclidean(lcm, components[i]) / (components[i] * lcm)
        
        record['target'] = lcm
                    
        return record
    
    @staticmethod
    def log(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = math.log(components[0], 
                                          components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def abssub(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = abs(components[0] - components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def eq(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] == components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def ne(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] != components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def gt(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] > components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def lt(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] < components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def ge(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] >= components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def le(num_operand, i) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [num for num in range (i, i+num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] <= components[1])
            
            return record
        
        else : 
            return None
