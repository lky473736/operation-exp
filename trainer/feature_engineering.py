import numpy as np
import pandas as pd

def extract_safe_features(operand1, operand2, result):
    """
    Extract completely safe features without any data leakage
    
    Args:
        operand1, operand2, result: arrays or single values
    
    Returns:
        feature_array: safe enhanced features
    """
    # Ensure arrays
    op1 = np.array(operand1, dtype=float)
    op2 = np.array(operand2, dtype=float) 
    res = np.array(result, dtype=float)
    
    # Basic features
    features = [op1, op2, res]
    
    # SAFE: Pure differences (no operation hints)
    features.extend([
        res - op1,                    # how different result is from op1
        res - op2,                    # how different result is from op2
        op1 - op2,                    # difference between operands
        abs(res - op1),              # absolute difference from op1
        abs(res - op2),              # absolute difference from op2
        abs(op1 - op2),              # absolute difference between operands
    ])
    
    # SAFE: Ratios (with epsilon to avoid division by zero)
    eps = 1e-8
    features.extend([
        res / (op1 + eps),           # ratio of result to op1
        res / (op2 + eps),           # ratio of result to op2
        op1 / (op2 + eps),           # ratio between operands
        op2 / (op1 + eps),           # inverse ratio between operands
    ])
    
    # SAFE: Relative comparisons (no specific operation results)
    features.extend([
        (res > op1).astype(float),           # result greater than op1?
        (res > op2).astype(float),           # result greater than op2?
        (res < op1).astype(float),           # result less than op1?
        (res < op2).astype(float),           # result less than op2?
        (res > 0).astype(float),             # result is positive?
        (res < 0).astype(float),             # result is negative?
        (op1 > op2).astype(float),           # op1 greater than op2?
        (op1 == op2).astype(float),          # operands are equal?
    ])
    
    # SAFE: Magnitude and scale relationships
    features.extend([
        abs(res),                            # magnitude of result
        abs(op1),                            # magnitude of op1
        abs(op2),                            # magnitude of op2
        abs(res) / (abs(op1) + abs(op2) + eps),  # result scale relative to operands
    ])
    
    # SAFE: Sign information
    features.extend([
        np.sign(res),                        # sign of result (-1, 0, 1)
        np.sign(op1),                        # sign of op1
        np.sign(op2),                        # sign of op2
        (np.sign(res) == np.sign(op1)).astype(float),  # same sign as op1?
        (np.sign(res) == np.sign(op2)).astype(float),  # same sign as op2?
    ])
    
    # SAFE: Logarithmic relationships (magnitude only)
    with np.errstate(divide='ignore', invalid='ignore'):
        features.extend([
            np.log(abs(res) + eps),          # log of result magnitude
            np.log(abs(op1) + eps),          # log of op1 magnitude  
            np.log(abs(op2) + eps),          # log of op2 magnitude
        ])
    
    # SAFE: Type indicators (integer vs float)
    features.extend([
        (op1 == np.floor(op1)).astype(float),    # op1 is integer?
        (op2 == np.floor(op2)).astype(float),    # op2 is integer?
        (res == np.floor(res)).astype(float),    # result is integer?
    ])
    
    # SAFE: Range and bound information
    features.extend([
        (abs(res) > abs(op1)).astype(float),     # result magnitude > op1 magnitude?
        (abs(res) > abs(op2)).astype(float),     # result magnitude > op2 magnitude?
        (abs(res) < abs(op1)).astype(float),     # result magnitude < op1 magnitude?
        (abs(res) < abs(op2)).astype(float),     # result magnitude < op2 magnitude?
    ])
    
    # SAFE: Reciprocal and power relationships (general)
    with np.errstate(divide='ignore', invalid='ignore'):
        features.extend([
            1 / (abs(res) + eps),            # reciprocal of result magnitude
            1 / (abs(op1) + eps),            # reciprocal of op1 magnitude
            1 / (abs(op2) + eps),            # reciprocal of op2 magnitude
        ])
    
    # SAFE: Statistical relationships
    operand_mean = (op1 + op2) / 2
    features.extend([
        res - operand_mean,                  # difference from operand mean
        abs(res - operand_mean),             # absolute difference from operand mean
        abs(op1 - operand_mean),             # op1 distance from mean
        abs(op2 - operand_mean),             # op2 distance from mean
    ])
    
    # Stack all features
    features_array = np.column_stack(features)
    
    # Replace any inf or nan values
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return features_array

def apply_feature_engineering(X, use_feature_engineering=True):
    """
    Apply safe feature engineering to input data
    
    Args:
        X: input array [operand1, operand2, result]
        use_feature_engineering: whether to apply feature engineering
    
    Returns:
        processed features
    """
    if not use_feature_engineering:
        return X
    
    # Extract operands and result
    operand1 = X[:, 0]
    operand2 = X[:, 1] 
    result = X[:, 2]
    
    # Apply safe feature engineering
    enhanced_features = extract_safe_features(operand1, operand2, result)
    
    return enhanced_features

def get_feature_names():
    """Return safe feature names for interpretation"""
    return [
        'operand1', 'operand2', 'result',
        'res_minus_op1', 'res_minus_op2', 'op1_minus_op2',
        'abs_res_minus_op1', 'abs_res_minus_op2', 'abs_op1_minus_op2',
        'res_div_op1', 'res_div_op2', 'op1_div_op2', 'op2_div_op1',
        'res_gt_op1', 'res_gt_op2', 'res_lt_op1', 'res_lt_op2', 
        'res_positive', 'res_negative', 'op1_gt_op2', 'op1_eq_op2',
        'abs_res', 'abs_op1', 'abs_op2', 'res_scale',
        'sign_res', 'sign_op1', 'sign_op2', 'same_sign_op1', 'same_sign_op2',
        'log_abs_res', 'log_abs_op1', 'log_abs_op2',
        'op1_is_int', 'op2_is_int', 'res_is_int',
        'abs_res_gt_abs_op1', 'abs_res_gt_abs_op2', 'abs_res_lt_abs_op1', 'abs_res_lt_abs_op2',
        'recip_res', 'recip_op1', 'recip_op2',
        'res_minus_mean', 'abs_res_minus_mean', 'abs_op1_minus_mean', 'abs_op2_minus_mean'
    ]