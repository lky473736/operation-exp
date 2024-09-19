'''
    operation - Gyuyeon Lim (lky473736)
    tool/operator
'''

import math
import numpy as np
import pandas as pd
import random

class Operator (object) :
    operators = {
        '+': ['add', lambda num_operand: Operator.add(num_operand)],
        '-': ['sub', lambda num_operand: Operator.sub(num_operand)],
        '*': ['mul', lambda num_operand: Operator.mul(num_operand)],
        '/': ['truediv', lambda num_operand: Operator.truediv(num_operand)],
        '%': ['mod', lambda num_operand: Operator.mod(num_operand)],
        '**': ['pow', lambda num_operand: Operator.pow(num_operand)],
        '//': ['floordiv', lambda num_operand: Operator.floordiv(num_operand)],

        '==': ['eq', lambda num_operand: Operator.eq(num_operand)],
        '!=': ['ne', lambda num_operand: Operator.ne(num_operand)],
        '>': ['gt', lambda num_operand: Operator.gt(num_operand)],
        '<': ['lt', lambda num_operand: Operator.lt(num_operand)],
        '>=': ['ge', lambda num_operand: Operator.ge(num_operand)],
        '<=': ['le', lambda num_operand: Operator.le(num_operand)],

        '&': ['bit_and', lambda num_operand: Operator.bit_and(num_operand)],
        '|': ['bit_or', lambda num_operand: Operator.bit_or(num_operand)],
        '^': ['bit_xor', lambda num_operand: Operator.bit_xor(num_operand)],
        '<<': ['lshift', lambda num_operand: Operator.lshift(num_operand)],
        '>>': ['rshift', lambda num_operand: Operator.rshift(num_operand)],

        'divmod': ['divmod', lambda num_operand: Operator.divmod(num_operand)],
        'max': ['max', lambda num_operand: Operator.max(num_operand)],
        'min': ['min', lambda num_operand: Operator.min(num_operand)],
        'hypot': ['hypot', lambda num_operand: Operator.hypot(num_operand)],

        'gcd': ['gcd', lambda num_operand: Operator.gcd(num_operand)],
        'lcm': ['lcm', lambda num_operand: Operator.lcm(num_operand)],
        'log': ['log', lambda num_operand: Operator.log(num_operand)],
        'abssub': ['abssub', lambda num_operand: Operator.abssub(num_operand)]
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
    def add(num_operand) :
        feature, target = Operator.montage_record(num_operand)
        components = [random.randint(0, 101) for i in range (num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]
            
        record['target'] = sum(components)
        
        return record
    
    @staticmethod
    def sub(num_operand) :
        feature, target = Operator.montage_record(num_operand)
        components = [random.randint(0, 101) for i in range (num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]
        
        for i in range (num_operand - 1)  :
            target = components[i] - components[i+1]

        record['target'] = target
        
        return record
    
    @staticmethod
    def mul(num_operand) :
        feature, target = Operator.montage_record(num_operand)
        components = [random.randint(0, 101) for i in range (num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]
        
        target = 1
        for i in range (num_operand)  :
            target = target * components[i]

        record['target'] = target
        
        return record
    
    @staticmethod
    def truediv(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            try : 
                record['target'] = components[0] / components[1]
            
            except ZeroDivisionError : 
                record['target'] = None
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def floordiv(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            try : 
                record['target'] = components[0] // components[1]
            
            except ZeroDivisionError : 
                record['target'] = None
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def mod(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            try : 
                record['target'] = components[0] % components[1]
            
            except ZeroDivisionError : 
                record['target'] = None
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def pow(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] ** components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def bit_and(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] & components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def bit_or(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] | components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def bit_xor(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] ^ components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def lshift(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] << components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def rshift(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = components[0] >> components[1]
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def divmod(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            try : 
                record['target'] = sum(divmod(components[0], components[1])) # divmod's target is the sumation share and remainder
            
            except ZeroDivisionError : 
                record['target'] = None

            return record
        
        else : 
            return None
    
    @staticmethod
    def max(num_operand) :
        feature, target = Operator.montage_record(num_operand)
        components = [random.randint(0, 101) for i in range (num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]

        record['target'] = max(components)
        
        return record
    
    @staticmethod
    def min(num_operand) :
        feature, target = Operator.montage_record(num_operand)
        components = [random.randint(0, 101) for i in range (num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]

        record['target'] = min(components)
        
        return record
    
    @staticmethod
    def hypot(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = math.hypot(components[0], 
                                          components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def gcd(num_operand) :
        feature, target = Operator.montage_record(num_operand)
        components = [random.randint(0, 101) for i in range (num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]

        gcd = components[0]
    
        for i in range (num_operand) : 
            if components[i] == 0 : 
                gcd = None
                break
            
            gcd = Operator.euclidean(gcd, components[i])
        
        record['target'] = gcd
                    
        return record
    
    @staticmethod
    def lcm(num_operand) :
        feature, target = Operator.montage_record(num_operand)
        components = [random.randint(0, 101) for i in range (num_operand)]
        
        record = dict()
        
        for i in range (num_operand) : 
            record[feature[i]] = components[i]

        lcm = components[0]
        for i in range (num_operand) : 
            if components[i] == 0 : 
                lcm = None
                break
            
            lcm = Operator.euclidean(lcm, components[i]) / (components[i] * lcm)
        
        record['target'] = lcm
                    
        return record
    
    @staticmethod
    def log(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            try : 
                record['target'] = math.log(components[0], components[1])
                
            except (ValueError, ZeroDivisionError) : 
                record['target'] = None
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def abssub(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = abs(components[0] - components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def eq(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] == components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def ne(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] != components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def gt(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] > components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def lt(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] < components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def ge(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] >= components[1])
            
            return record
        
        else : 
            return None
    
    @staticmethod
    def le(num_operand) :
        if num_operand == 2 : 
            feature, target = Operator.montage_record(num_operand)
            components = [random.randint(0, 101) for i in range (num_operand)]
            
            record = dict()
            
            for i in range (num_operand) : 
                record[feature[i]] = components[i]
            
            record['target'] = (components[0] <= components[1])
            
            return record
        
        else : 
            return None
