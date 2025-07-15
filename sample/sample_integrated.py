#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import warnings
import itertools
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available, using CPU")

class TabNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=32, n_a=32, n_steps=3, gamma=1.3):
        super(TabNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        self.attention_layers = nn.ModuleList()
        self.transform_layers = nn.ModuleList()
        self.decision_layers = nn.ModuleList()
        
        for step in range(n_steps):
            self.attention_layers.append(nn.Sequential(
                nn.Linear(input_dim, n_a),
                nn.BatchNorm1d(n_a),
                nn.ReLU(),
                nn.Linear(n_a, input_dim)
            ))
            
            self.transform_layers.append(nn.Sequential(
                nn.Linear(input_dim, n_d + n_a),
                nn.BatchNorm1d(n_d + n_a),
                nn.ReLU()
            ))
            
            self.decision_layers.append(nn.Linear(n_d + n_a, n_d))
        
        self.final_layer = nn.Linear(n_d, output_dim)
    
    def forward(self, x):
        x = self.initial_bn(x)
        prior = torch.ones_like(x)
        decision_outputs = []
        
        for step in range(self.n_steps):
            mask_logits = self.attention_layers[step](x)
            mask_logits = mask_logits * prior
            mask = F.softmax(mask_logits, dim=1)
            
            masked_features = x * mask
            transformed = self.transform_layers[step](masked_features)
            decision_out = F.relu(self.decision_layers[step](transformed))
            decision_outputs.append(decision_out)
            
            prior = prior * (self.gamma - mask)
        
        decision_sum = sum(decision_outputs)
        output = self.final_layer(decision_sum)
        return output

class FCNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(FCNClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class RelationalOperatorNet(nn.Module):
    def __init__(self, input_dim, output_dim, relation_dim=64, fusion_dim=128, dropout_rate=0.3):
        super(RelationalOperatorNet, self).__init__()
        self.input_dim = input_dim
        self.n_operands = input_dim - 1
        self.relation_dim = relation_dim
        self.fusion_dim = fusion_dim
        
        n_pairs = self.n_operands * (self.n_operands - 1) // 2
        self.pairwise_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, relation_dim),
                nn.ReLU(),
                nn.BatchNorm1d(relation_dim)
            ) for _ in range(n_pairs)
        ])
        
        self.result_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4, relation_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(relation_dim // 2)
            ) for _ in range(self.n_operands)
        ])
        
        pairwise_total_dim = n_pairs * relation_dim
        result_total_dim = self.n_operands * (relation_dim // 2)
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(pairwise_total_dim + result_total_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, output_dim)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        operands = x[:, :-1]
        result = x[:, -1:]
        
        pairwise_features = []
        pair_idx = 0
        for i in range(self.n_operands):
            for j in range(i + 1, self.n_operands):
                op_i = operands[:, i:i+1]
                op_j = operands[:, j:j+1]
                
                diff = op_i - op_j
                ratio = op_i / (op_j + 1e-8)
                product = op_i * op_j
                
                pair_features = torch.cat([diff, ratio, product], dim=1)
                pair_out = self.pairwise_layers[pair_idx](pair_features)
                pairwise_features.append(pair_out)
                pair_idx += 1
        
        result_features = []
        for i in range(self.n_operands):
            operand = operands[:, i:i+1]
            diff = result - operand
            ratio = result / (operand + 1e-8)
            
            result_input = torch.cat([diff, ratio, result, operand], dim=1)
            result_out = self.result_layers[i](result_input)
            result_features.append(result_out)
        
        all_pairwise = torch.cat(pairwise_features, dim=1)
        all_result = torch.cat(result_features, dim=1)
        
        combined = torch.cat([all_pairwise, all_result], dim=1)
        output = self.fusion_layers(combined)
        
        return output

def get_user_inputs():
    print("OPERATOR SEQUENCE CLASSIFICATION CONFIGURATION")
    print("="*60)
    
    while True:
        try:
            n_operands = int(input("Number of operands (2-6): "))
            if 2 <= n_operands <= 6:
                break
            else:
                print("Please enter a number between 2 and 6")
        except ValueError:
            print("Please enter a valid integer")
    
    while True:
        try:
            samples_per_class = int(input("Samples per class (100-10000): "))
            if 100 <= samples_per_class <= 10000:
                break
            else:
                print("Please enter a number between 100 and 10000")
        except ValueError:
            print("Please enter a valid integer")
    
    print("\nScaling options:")
    print("1. StandardScaler (z-score normalization)")
    print("2. MinMaxScaler")
    while True:
        scaler_choice = input("Choose scaler (1 or 2): ").strip()
        if scaler_choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    print("\nAvailable operators:")
    all_operators = ['add', 'sub', 'mul', 'truediv', 'mod', 'floordiv', 'divmod', 'max', 'min', 'hypot', 'gcd', 'lcm', 'log', 'abssub']
    for i, op in enumerate(all_operators, 1):
        print(f"{i:2d}. {op}")
    
    print("Enter operator numbers separated by spaces (or press Enter for all):")
    operator_input = input().strip()
    
    if operator_input:
        try:
            operator_indices = [int(x)-1 for x in operator_input.split()]
            selected_operators = [all_operators[i] for i in operator_indices if 0 <= i < len(all_operators)]
        except:
            print("Invalid input, using all operators")
            selected_operators = all_operators
    else:
        selected_operators = all_operators
    
    while True:
        try:
            batch_size = int(input("Batch size (16-512): "))
            if 16 <= batch_size <= 512:
                break
            else:
                print("Please enter a number between 16 and 512")
        except ValueError:
            print("Please enter a valid integer")
    
    while True:
        try:
            epochs = int(input("Number of epochs (10-300): "))
            if 10 <= epochs <= 300:
                break
            else:
                print("Please enter a number between 10 and 300")
        except ValueError:
            print("Please enter a valid integer")
    
    while True:
        try:
            learning_rate = float(input("Learning rate (0.0001-0.1): "))
            if 0.0001 <= learning_rate <= 0.1:
                break
            else:
                print("Please enter a number between 0.0001 and 0.1")
        except ValueError:
            print("Please enter a valid number")
    
    print("\nOptimizer options:")
    optimizers = ['adam', 'sgd', 'rmsprop', 'adamw']
    for i, opt in enumerate(optimizers, 1):
        print(f"{i}. {opt}")
    while True:
        opt_choice = input("Choose optimizer (1-4): ").strip()
        if opt_choice in ['1', '2', '3', '4']:
            optimizer = optimizers[int(opt_choice)-1]
            break
        print("Please enter a number between 1 and 4")
    
    print("\nModel options:")
    models = ['TabNet', 'FCN', 'RelationalNet']
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    while True:
        model_choice = input("Choose model (1-3): ").strip()
        if model_choice in ['1', '2', '3']:
            selected_model = models[int(model_choice)-1]
            break
        print("Please enter a number between 1 and 3")
    
    config = {
        'n_operands': n_operands,
        'samples_per_class': samples_per_class,
        'scaler_type': 'standard' if scaler_choice == '1' else 'minmax',
        'operators': selected_operators,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'optimizer': optimizer,
        'model': selected_model
    }
    
    return config

def get_operator_function(op_name):
    operators = {
        'add': lambda x, y: round(x + y, 3),
        'sub': lambda x, y: round(x - y, 3),
        'mul': lambda x, y: round(x * y, 3),
        'truediv': lambda x, y: round(x / (y + 1e-8), 3),
        'mod': lambda x, y: round(np.mod(x, y + 1e-8), 3),
        'floordiv': lambda x, y: round(np.floor(x / (y + 1e-8)), 3),
        'divmod': lambda x, y: round((x // (y + 1e-8)) + (x % (y + 1e-8)), 3),
        'max': lambda x, y: round(np.maximum(x, y), 3),
        'min': lambda x, y: round(np.minimum(x, y), 3),
        'hypot': lambda x, y: round(np.sqrt(x**2 + y**2), 3),
        'gcd': lambda x, y: round(np.gcd(np.abs(x).astype(int), np.abs(y).astype(int)), 3),
        'lcm': lambda x, y: round(np.lcm(np.abs(x).astype(int), np.abs(y).astype(int)), 3),
        'log': lambda x, y: round(np.log(np.abs(y) + 1e-8) / np.log(np.abs(x) + 1e-8), 3),
        'abssub': lambda x, y: round(np.abs(x - y), 3)
    }
    return operators.get(op_name, operators['add'])

def generate_operator_sequence_data(config):
    n_operands = config['n_operands']
    samples_per_class = config['samples_per_class']
    operators = config['operators']
    
    operator_sequences = list(itertools.product(operators, repeat=n_operands-1))
    
    data = []
    labels = []
    
    print(f"Generating {samples_per_class} samples per class...")
    print(f"Total classes: {len(operator_sequences)}")
    print(f"Total samples: {len(operator_sequences) * samples_per_class}")
    
    for seq_idx, seq in enumerate(operator_sequences):
        operator_funcs = [get_operator_function(op) for op in seq]
        
        samples_generated = 0
        attempts = 0
        max_attempts = samples_per_class * 10
        
        while samples_generated < samples_per_class and attempts < max_attempts:
            attempts += 1
            
            operands = np.random.uniform(-10, 10, n_operands)
            
            # Special handling for certain operators
            if any(op in ['gcd', 'lcm'] for op in seq):
                operands = np.random.randint(1, 20, n_operands).astype(float)
            elif any(op in ['truediv', 'floordiv', 'mod', 'divmod', 'log'] for op in seq):
                for i in range(n_operands):
                    if abs(operands[i]) < 0.1:
                        operands[i] = np.sign(operands[i]) * (0.1 + abs(operands[i]))
            
            try:
                result = operands[0]
                for i, op_func in enumerate(operator_funcs):
                    if seq[i] in ['gcd', 'lcm']:
                        # Ensure positive integers for gcd/lcm
                        op1 = max(1, int(abs(result)))
                        op2 = max(1, int(abs(operands[i+1])))
                        result = round(float(op_func(op1, op2)), 3)
                    elif seq[i] == 'log':
                        # Ensure positive values for log
                        base = max(1.1, abs(result))
                        value = max(0.1, abs(operands[i+1]))
                        result = round(op_func(base, value), 3)
                    elif seq[i] == 'hypot':
                        result = round(op_func(result, operands[i+1]), 3)
                    elif seq[i] == 'divmod':
                        # For divmod, we take quotient + remainder
                        if abs(operands[i+1]) > 0.1:
                            result = round(op_func(result, operands[i+1]), 3)
                        else:
                            continue
                    else:
                        result = round(op_func(result, operands[i+1]), 3)
                
                if np.isfinite(result) and abs(result) < 1e6:
                    sample = list(operands) + [result]
                    data.append(sample)
                    labels.append('_'.join(seq))
                    samples_generated += 1
                    
            except:
                continue
        
        if seq_idx % max(1, len(operator_sequences)//10) == 0:
            print(f"  Generated {seq_idx + 1}/{len(operator_sequences)} classes...")
    
    print(f"Successfully generated {len(data)} samples")
    return np.array(data), np.array(labels), operator_sequences

def get_feature_info(config, use_feature_engineering=False):
    n_operands = config['n_operands']
    
    if use_feature_engineering:
        feature_names = [f'operand{i+1}' for i in range(n_operands)] + ['result']
        feature_types = ['Basic'] * (n_operands + 1)
        
        # Result-operand relationships
        for i in range(n_operands):
            feature_names.extend([
                f'res_minus_op{i+1}', f'abs_res_minus_op{i+1}', f'res_div_op{i+1}',
                f'res_gt_op{i+1}', f'res_lt_op{i+1}', f'same_sign_res_op{i+1}'
            ])
            feature_types.extend(['Result_Diff', 'Result_AbsDiff', 'Result_Ratio', 
                                'Result_Comp', 'Result_Comp', 'Result_Sign'] )
        
        # Pairwise operand relationships
        for i in range(n_operands):
            for j in range(i+1, n_operands):
                feature_names.extend([
                    f'op{i+1}_minus_op{j+1}', f'abs_op{i+1}_minus_op{j+1}',
                    f'op{i+1}_div_op{j+1}', f'op{j+1}_div_op{i+1}',
                    f'op{i+1}_gt_op{j+1}', f'op{i+1}_eq_op{j+1}'
                ])
                feature_types.extend(['Pairwise_Diff', 'Pairwise_AbsDiff', 'Pairwise_Ratio', 
                                    'Pairwise_Ratio', 'Pairwise_Comp', 'Pairwise_Comp'])
        
        # Global properties
        feature_names.extend(['res_positive', 'res_negative', 'abs_res', 'sign_res'])
        feature_types.extend(['Global_Sign', 'Global_Sign', 'Global_Mag', 'Global_Sign'])
        
        # Operand properties
        for i in range(n_operands):
            feature_names.extend([
                f'abs_op{i+1}', f'sign_op{i+1}', 
                f'abs_res_gt_abs_op{i+1}', f'abs_res_lt_abs_op{i+1}'
            ])
            feature_types.extend(['Magnitude', 'Sign', 'Magnitude_Comp', 'Magnitude_Comp'])
        
        # Scale relationships
        feature_names.extend([
            'res_scale_sum', 'res_minus_mean', 'abs_res_minus_mean',
            'abs_res_gt_max', 'abs_res_lt_min'
        ])
        feature_types.extend(['Scale', 'Scale', 'Scale', 'Scale', 'Scale'])
        
        # Logarithmic features
        feature_names.append('log_abs_res')
        feature_types.append('Logarithmic')
        for i in range(n_operands):
            feature_names.append(f'log_abs_op{i+1}')
            feature_types.append('Logarithmic')
        
        # Type indicators
        feature_names.append('res_is_int')
        feature_types.append('Type')
        for i in range(n_operands):
            feature_names.append(f'op{i+1}_is_int')
            feature_types.append('Type')
        
        # Reciprocal features
        feature_names.append('recip_res')
        feature_types.append('Reciprocal')
        for i in range(n_operands):
            feature_names.append(f'recip_op{i+1}')
            feature_types.append('Reciprocal')
        
        # Statistical features
        for i in range(n_operands):
            feature_names.extend([f'abs_op{i+1}_minus_mean', f'op{i+1}_gt_mean'])
            feature_types.extend(['Statistical', 'Statistical'])
        
    else:
        feature_names = [f'operand{i+1}' for i in range(n_operands)] + ['result']
        feature_types = ['Basic'] * len(feature_names)
    
    return feature_names, feature_types

def extract_safe_features(operands, result):
    """
    Extract completely safe features without any data leakage for n operands
    
    Args:
        operands: array of operands [batch_size, n_operands]
        result: array of results [batch_size, 1]
    
    Returns:
        feature_array: safe enhanced features
    """
    operands = np.array(operands, dtype=float)
    result = np.array(result, dtype=float).reshape(-1, 1)
    n_operands = operands.shape[1]
    eps = 1e-8
    
    # Basic features
    features = [operands, result]
    
    # SAFE: Differences between result and each operand
    for i in range(n_operands):
        op = operands[:, i:i+1]
        features.extend([
            result - op,                    # how different result is from operand i
            np.abs(result - op),           # absolute difference from operand i
            result / (op + eps),           # ratio of result to operand i
            (result > op).astype(float),   # result greater than operand i?
            (result < op).astype(float),   # result less than operand i?
            (np.sign(result) == np.sign(op)).astype(float),  # same sign as operand i?
        ])
    
    # SAFE: Pairwise relationships between operands
    for i in range(n_operands):
        for j in range(i+1, n_operands):
            op_i = operands[:, i:i+1]
            op_j = operands[:, j:j+1]
            features.extend([
                op_i - op_j,                    # difference between operands
                np.abs(op_i - op_j),           # absolute difference between operands
                op_i / (op_j + eps),           # ratio between operands
                op_j / (op_i + eps),           # inverse ratio between operands
                (op_i > op_j).astype(float),   # operand i greater than operand j?
                (op_i == op_j).astype(float),  # operands are equal?
            ])
    
    # SAFE: Global properties
    features.extend([
        (result > 0).astype(float),             # result is positive?
        (result < 0).astype(float),             # result is negative?
        np.abs(result),                         # magnitude of result
        np.sign(result),                        # sign of result (-1, 0, 1)
    ])
    
    # SAFE: Operand magnitude and scale relationships
    for i in range(n_operands):
        op = operands[:, i:i+1]
        features.extend([
            np.abs(op),                         # magnitude of operand i
            np.sign(op),                        # sign of operand i
            (np.abs(result) > np.abs(op)).astype(float),  # result magnitude > operand i magnitude?
            (np.abs(result) < np.abs(op)).astype(float),  # result magnitude < operand i magnitude?
        ])
    
    # SAFE: Scale relative to all operands
    operand_sum = np.sum(np.abs(operands), axis=1, keepdims=True)
    operand_mean = np.mean(operands, axis=1, keepdims=True)
    operand_max = np.max(np.abs(operands), axis=1, keepdims=True)
    operand_min = np.min(np.abs(operands), axis=1, keepdims=True)
    
    features.extend([
        np.abs(result) / (operand_sum + eps),   # result scale relative to sum of operands
        result - operand_mean,                  # difference from operand mean
        np.abs(result - operand_mean),          # absolute difference from operand mean
        (np.abs(result) > operand_max).astype(float),  # result magnitude > max operand?
        (np.abs(result) < operand_min).astype(float),  # result magnitude < min operand?
    ])
    
    # SAFE: Logarithmic relationships (magnitude only)
    with np.errstate(divide='ignore', invalid='ignore'):
        features.append(np.log(np.abs(result) + eps))  # log of result magnitude
        for i in range(n_operands):
            op = operands[:, i:i+1]
            features.append(np.log(np.abs(op) + eps))   # log of operand i magnitude
    
    # SAFE: Type indicators (integer vs float)
    features.append((result == np.floor(result)).astype(float))  # result is integer?
    for i in range(n_operands):
        op = operands[:, i:i+1]
        features.append((op == np.floor(op)).astype(float))      # operand i is integer?
    
    # SAFE: Reciprocal relationships (general)
    with np.errstate(divide='ignore', invalid='ignore'):
        features.append(1 / (np.abs(result) + eps))  # reciprocal of result magnitude
        for i in range(n_operands):
            op = operands[:, i:i+1]
            features.append(1 / (np.abs(op) + eps))   # reciprocal of operand i magnitude
    
    # SAFE: Statistical relationships
    for i in range(n_operands):
        op = operands[:, i:i+1]
        features.extend([
            np.abs(op - operand_mean),          # operand i distance from mean
            (op > operand_mean).astype(float),  # operand i greater than mean?
        ])
    
    # Stack all features
    features_array = np.column_stack(features)
    
    # Replace any inf or nan values
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return features_array

def extract_sequence_features(data, config):
    """Apply safe feature engineering to n-operand data"""
    operands = data[:, :-1]
    result = data[:, -1]
    
    enhanced_features = extract_safe_features(operands, result)
    return enhanced_features

def prepare_sequence_data(config, use_feature_engineering=False, X_raw=None, y_raw=None):
    print(f"Preparing sequence data with feature engineering: {use_feature_engineering}")
    
    if X_raw is None or y_raw is None:
        X, y, operator_sequences = generate_operator_sequence_data(config)
    else:
        X, y, operator_sequences = X_raw, y_raw, None
    
    if not use_feature_engineering:
        unique_sequences, sequence_counts = np.unique(y, return_counts=True)
        
        print(f"\nOPERATOR SEQUENCE INFORMATION")
        print("-" * 50)
        print(f"Total number of sequence classes: {len(unique_sequences)}")
        print(f"Sequence format: operand1 ○ operand2 ● ... = result")
        print(f"Available operators: {', '.join(config['operators'])}")
        print(f"Total possible sequences: {len(unique_sequences)}")
        print(f"Total samples: {len(y)}")
        
        print(f"\nClass balance check:")
        print(f"  Min samples per class: {np.min(sequence_counts)}")
        print(f"  Max samples per class: {np.max(sequence_counts)}")
        print(f"  Mean samples per class: {np.mean(sequence_counts):.1f}")
        print(f"  Std samples per class: {np.std(sequence_counts):.1f}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    feature_names, feature_types = get_feature_info(config, use_feature_engineering)
    
    print(f"\nFEATURE INFORMATION")
    print("-" * 40)
    
    if use_feature_engineering:
        X = extract_sequence_features(X, config)
        print(f"Features expanded from {config['n_operands']+1} to {X.shape[1]} dimensions")
        
        feature_type_counts = {}
        for ftype in feature_types:
            feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1
        
        print("Feature types breakdown:")
        for ftype, count in feature_type_counts.items():
            print(f"  {ftype:>15}: {count:>2} features")
            
    else:
        print(f"Using basic features: {X.shape[1]} dimensions")
        print("Feature names:")
        for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
            print(f"  [{i:2d}] {name:>15} ({ftype})")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    if config['scaler_type'] == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"\nDATA SPLIT INFORMATION")
    print("-" * 40)
    print(f"Training set:   {X_train.shape[0]:>6} samples × {X_train.shape[1]:>2} features")
    print(f"Validation set: {X_val.shape[0]:>6} samples × {X_val.shape[1]:>2} features")
    print(f"Test set:       {X_test.shape[0]:>6} samples × {X_test.shape[1]:>2} features")
    
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    print(f"\nClass balance in test set:")
    print(f"  Min: {np.min(test_counts)}, Max: {np.max(test_counts)}, Mean: {np.mean(test_counts):.1f}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test), le, scaler, (X, y, operator_sequences)

def get_optimizer(model_params, config):
    lr = config['learning_rate']
    opt_name = config['optimizer']
    
    if opt_name == 'adam':
        return optim.Adam(model_params, lr=lr)
    elif opt_name == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr)
    elif opt_name == 'adamw':
        return optim.AdamW(model_params, lr=lr)
    else:
        return optim.Adam(model_params, lr=lr)

def train_sequence_model(data, config, model_name):
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    print(f"\nTraining {model_name}")
    print("-" * 50)
    
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    
    if config['model'] == 'TabNet':
        model = TabNet(input_dim, output_dim, n_d=64, n_a=64, n_steps=4)
    elif config['model'] == 'FCN':
        model = FCNClassifier(input_dim, output_dim)
    elif config['model'] == 'RelationalNet':
        model = RelationalOperatorNet(input_dim, output_dim)
    
    model = model.to(device)
    
    print(f"Model architecture: {input_dim} → {config['model']} → {output_dim} classes")
    
    optimizer = get_optimizer(model.parameters(), config)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
    
    y_pred_classes = np.array(test_predictions)
    y_test_actual = np.array(test_targets)
    
    accuracy = accuracy_score(y_test_actual, y_pred_classes)
    precision = precision_score(y_test_actual, y_pred_classes, average='weighted')
    recall = recall_score(y_test_actual, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_actual, y_pred_classes, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_loss': best_val_loss
    }
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return model, history, y_pred_classes, metrics

def plot_sequence_training_curves(histories, model_names):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Operator Sequence Classification: Training Curves', fontsize=16, fontweight='bold')
    
    for i, (history, name) in enumerate(zip(histories, model_names)):
        color = ['blue', 'red'][i]
        
        axes[0, 0].plot(history['loss'], label=f'{name} Train', color=color, linewidth=2)
        axes[0, 0].plot(history['val_loss'], label=f'{name} Val', color=color, linestyle='--', linewidth=2)
        
        axes[0, 1].plot(history['accuracy'], label=f'{name} Train', color=color, linewidth=2)
        axes[0, 1].plot(history['val_accuracy'], label=f'{name} Val', color=color, linestyle='--', linewidth=2)
    
    axes[0, 0].set_title('Loss per Epoch', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Accuracy per Epoch', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    final_metrics = pd.DataFrame({
        'Original': [histories[0]['val_accuracy'][-1], histories[0]['val_loss'][-1]],
        'Feature Eng': [histories[1]['val_accuracy'][-1], histories[1]['val_loss'][-1]]
    }, index=['Val Accuracy', 'Val Loss'])
    
    axes[1, 0].bar(final_metrics.columns, final_metrics.loc['Val Accuracy'], 
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 0].set_title('Final Validation Accuracy', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    for i, v in enumerate(final_metrics.loc['Val Accuracy']):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    axes[1, 1].bar(final_metrics.columns, final_metrics.loc['Val Loss'], 
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title('Final Validation Loss', fontweight='bold')
    axes[1, 1].set_ylabel('Loss')
    for i, v in enumerate(final_metrics.loc['Val Loss']):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sequence_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sequence_confusion_matrices(y_test, y_pred_baseline, y_pred_fe, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    all_classes = class_names
    n_classes = len(all_classes)
    
    print(f"Creating confusion matrices for all {n_classes} classes...")
    
    cm_baseline = confusion_matrix(y_test, y_pred_baseline, labels=range(n_classes))
    
    sns.heatmap(cm_baseline, annot=False, cmap='Blues', fmt='d',
                xticklabels=all_classes, yticklabels=all_classes, ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_title('Original Model\nConfusion Matrix', 
                     fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Predicted Sequence', fontsize=12)
    axes[0].set_ylabel('True Sequence', fontsize=12)
    
    axes[0].set_xticklabels(all_classes, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(all_classes, rotation=0, fontsize=8)
    
    cm_fe = confusion_matrix(y_test, y_pred_fe, labels=range(n_classes))
    
    sns.heatmap(cm_fe, annot=False, cmap='Reds', fmt='d',
                xticklabels=all_classes, yticklabels=all_classes, ax=axes[1],
                cbar_kws={'label': 'Count'})
    axes[1].set_title('Feature Engineering Model\nConfusion Matrix', 
                     fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Predicted Sequence', fontsize=12)
    axes[1].set_ylabel('True Sequence', fontsize=12)
    
    axes[1].set_xticklabels(all_classes, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(all_classes, rotation=0, fontsize=8)
    
    diag_baseline = np.diag(cm_baseline)
    diag_fe = np.diag(cm_fe)
    total_baseline = np.sum(cm_baseline, axis=1)
    total_fe = np.sum(cm_fe, axis=1)
    
    acc_baseline = diag_baseline / (total_baseline + 1e-8)
    acc_fe = diag_fe / (total_fe + 1e-8)
    
    overall_acc_baseline = np.sum(diag_baseline) / np.sum(cm_baseline)
    overall_acc_fe = np.sum(diag_fe) / np.sum(cm_fe)
    
    axes[0].text(0.02, 0.98, f'Overall Accuracy: {overall_acc_baseline:.3f}', 
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1].text(0.02, 0.98, f'Overall Accuracy: {overall_acc_fe:.3f}', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('sequence_confusion_matrices_all_classes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    class_performance = pd.DataFrame({
        'Class': all_classes,
        'Original_Acc': acc_baseline,
        'FeatEng_Acc': acc_fe,
        'Improvement': acc_fe - acc_baseline
    })
    
    class_performance_sorted = class_performance.sort_values('FeatEng_Acc', ascending=False)
    
    print("TOP 10 BEST PERFORMING CLASSES (Feature Engineering):")
    print(class_performance_sorted.head(10)[['Class', 'Original_Acc', 'FeatEng_Acc', 'Improvement']].to_string(index=False, float_format='%.3f'))
    
    print("\nBOTTOM 10 CLASSES (Feature Engineering):")
    print(class_performance_sorted.tail(10)[['Class', 'Original_Acc', 'FeatEng_Acc', 'Improvement']].to_string(index=False, float_format='%.3f'))
    
    most_improved = class_performance.sort_values('Improvement', ascending=False)
    print("\nMOST IMPROVED CLASSES:")
    print(most_improved.head(10)[['Class', 'Original_Acc', 'FeatEng_Acc', 'Improvement']].to_string(index=False, float_format='%.3f'))
    
    return class_performance

def plot_sequence_metrics_comparison(metrics_baseline, metrics_fe):
    metrics_df = pd.DataFrame({
        'Original': [metrics_baseline['accuracy'], metrics_baseline['precision'], 
                    metrics_baseline['recall'], metrics_baseline['f1']],
        'Feature Engineering': [metrics_fe['accuracy'], metrics_fe['precision'], 
                               metrics_fe['recall'], metrics_fe['f1']]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=ax, color=['blue', 'red'], alpha=0.7, width=0.8)
    ax.set_title('Operator Sequence Classification: Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(title='Model Type')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    for i, (idx, row) in enumerate(metrics_df.iterrows()):
        for j, value in enumerate(row):
            ax.text(i + (j-0.5)*0.4, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sequence_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sequence_tsne_visualization(X_test, y_test, y_pred_baseline, y_pred_fe, class_names):
    print("Computing t-SNE embeddings for sequence data...")
    
    max_samples = 1000
    if len(X_test) > max_samples:
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
        y_pred_baseline = y_pred_baseline[indices]
        y_pred_fe = y_pred_fe[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, 
                             cmap='tab20', alpha=0.7, s=20)
    axes[0].set_title('True Operator Sequences', fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_baseline, 
                             cmap='tab20', alpha=0.7, s=20)
    axes[1].set_title('Original Model Predictions', fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    scatter = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_fe, 
                             cmap='tab20', alpha=0.7, s=20)
    axes[2].set_title('Feature Engineering Predictions', fontweight='bold')
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')
    
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal', 
                       pad=0.1, aspect=30, shrink=0.8)
    cbar.set_label('Operator Sequence Class')
    
    plt.tight_layout()
    plt.savefig('sequence_tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_sequence_classification_reports(y_test, y_pred_baseline, y_pred_fe, class_names):
    print("\n" + "="*80)
    print("DETAILED SEQUENCE CLASSIFICATION REPORTS")
    print("="*80)
    
    sample_classes = class_names[:15] if len(class_names) > 15 else class_names
    
    print(f"\nORIGINAL MODEL (showing first {len(sample_classes)} classes):")
    print("-" * 40)
    mask = np.isin(y_test, range(len(sample_classes)))
    print(classification_report(y_test[mask], y_pred_baseline[mask], 
                               target_names=sample_classes, zero_division=0))
    
    print(f"\nFEATURE ENGINEERING MODEL (showing first {len(sample_classes)} classes):")
    print("-" * 40)
    print(classification_report(y_test[mask], y_pred_fe[mask], 
                               target_names=sample_classes, zero_division=0))

def main():
    print("🎯 OPERATOR SEQUENCE CLASSIFICATION WITH CONFIGURABLE MODELS")
    print("="*70)
    print("operand1 ○ operand2 ● operand3 ... = result")
    print("Predict operator sequence [○, ●, ...]")
    print("="*70)
    
    config = get_user_inputs()
    
    print("\n🧪 EXPERIMENT OVERVIEW")
    print("-" * 50)
    print("📌 Task: Multi-class operator sequence classification")
    print(f"📌 Operands: {config['n_operands']}")
    print(f"📌 Operators: {', '.join(config['operators'])}")
    print(f"📌 Model: {config['model']}")
    print(f"📌 Samples per class: {config['samples_per_class']}")
    print(f"📌 Scaling: {config['scaler_type']}")
    print(f"📌 Optimizer: {config['optimizer']}")
    print(f"📌 Learning rate: {config['learning_rate']}")
    print(f"📌 Batch size: {config['batch_size']}")
    print(f"📌 Epochs: {config['epochs']}")
    
    all_histories = []
    all_models = []
    all_predictions = []
    all_metrics = []
    model_names = ['Original Features', 'Feature Engineering']
    
    print("\nGenerating shared balanced dataset...")
    X_raw, y_raw, operator_sequences = generate_operator_sequence_data(config)
    
    print("\n" + "="*60)
    print("PHASE 1: ORIGINAL FEATURES MODEL")
    print("="*60)
    
    data_baseline, le, scaler_baseline, _ = prepare_sequence_data(
        config, use_feature_engineering=False, X_raw=X_raw, y_raw=y_raw
    )
    
    model_baseline, history_baseline, y_pred_baseline, metrics_baseline = train_sequence_model(
        data_baseline, config, "Original Features"
    )
    
    all_histories.append(history_baseline)
    all_models.append(model_baseline)
    all_predictions.append(y_pred_baseline)
    all_metrics.append(metrics_baseline)
    
    print("\n" + "="*60)
    print("PHASE 2: FEATURE ENGINEERING MODEL")
    print("="*60)
    
    data_fe, le_fe, scaler_fe, _ = prepare_sequence_data(
        config, use_feature_engineering=True, X_raw=X_raw, y_raw=y_raw
    )
    
    model_fe, history_fe, y_pred_fe, metrics_fe = train_sequence_model(
        data_fe, config, "Feature Engineering"
    )
    
    all_histories.append(history_fe)
    all_models.append(model_fe)
    all_predictions.append(y_pred_fe)
    all_metrics.append(metrics_fe)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    print("\nPERFORMACE SUMMARY")
    print("-" * 50)
    print(f"{'Metric':<15} {'Original':<12} {'Feature Eng':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        baseline_val = metrics_baseline[metric]
        fe_val = metrics_fe[metric]
        improvement = ((fe_val - baseline_val) / baseline_val) * 100
        
        print(f"{metric.capitalize():<15} {baseline_val:<12.4f} {fe_val:<12.4f} {improvement:>+8.1f}%")
    
    winner = "Feature Engineering" if metrics_fe['f1'] > metrics_baseline['f1'] else "Original Features"
    improvement = abs(metrics_fe['f1'] - metrics_baseline['f1']) / max(metrics_fe['f1'], metrics_baseline['f1']) * 100
    
    print(f"\nWINNER: {winner} (F1 improvement: {improvement:.1f}%)")
    
    class_names = le.classes_
    
    print("\nGENERATING COMPREHENSIVE VISUALIZATIONS")
    print("-" * 50)
    
    print("Creating training curves comparison...")
    plot_sequence_training_curves(all_histories, model_names)
    
    print("Creating metrics comparison...")
    plot_sequence_metrics_comparison(metrics_baseline, metrics_fe)
    
    print("Creating confusion matrices...")
    class_performance = plot_sequence_confusion_matrices(
        data_baseline[5], y_pred_baseline, y_pred_fe, class_names
    )
    
    print("Creating t-SNE visualization...")
    plot_sequence_tsne_visualization(
        data_baseline[2], data_baseline[5], y_pred_baseline, y_pred_fe, class_names
    )
    
    print_sequence_classification_reports(
        data_baseline[5], y_pred_baseline, y_pred_fe, class_names
    )
    
    print("\n" + "="*80)
    print("KEY INSIGHTS & CONCLUSIONS")
    print("="*80)
    
    print("\nModel Performance:")
    if metrics_fe['f1'] > metrics_baseline['f1']:
        print("Feature engineering significantly improved sequence classification")
        print("Advanced features help distinguish between operator sequences")
        print("Pairwise relationships and result comparisons are crucial")
    else:
        print("Original features performed surprisingly well")
        print("Sequence patterns might be simpler than expected")
        print("Feature engineering may have introduced noise")
    
    print(f"\nClassification Complexity:")
    print(f"• Total sequences: {len(class_names)} classes")
    print(f"• Model used: {config['model']}")
    print(f"• Operands: {config['n_operands']}")
    
    print(f"\nRecommendations:")
    if metrics_fe['f1'] > metrics_baseline['f1']:
        print("• Use feature engineering for production sequence classification")
        print("• Focus on pairwise and result relationship features")
        print("• Consider ensemble methods for further improvement")
    else:
        print("• Simple features may be sufficient for this task")
        print("• Focus on data quality and model architecture instead")
        print("• Consider domain-specific feature engineering")
    
    print(f"\nExperiment completed successfully!")
    print(f"Best model: {winner} with F1-score: {max(metrics_baseline['f1'], metrics_fe['f1']):.4f}")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()
    
    print("\nAll analyses completed!")
    print("Generated files:")
    print("  • sequence_training_curves.png")
    print("  • sequence_metrics_comparison.png") 
    print("  • sequence_confusion_matrices_all_classes.png")
    print("  • sequence_tsne_visualization.png")