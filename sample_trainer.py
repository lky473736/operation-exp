#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath('.'))

from trainer.utils import run_experiment

def main():
    """Run sample experiments"""
    
    # # Example 1: Section 1 (Regression) - Addition operation
    # print("Running Section 1 experiment...")
    # run_experiment(
    #     data_path='data/sample/sample_add.csv',
    #     model_type='densenet',
    #     section=1,
    #     epochs=50,
    #     batch_size=32,
    #     # DenseNet parameters
    #     hidden_dims=[256, 128, 64],
    #     dropout_rate=0.2,
    #     l2_reg=0.001
    # )
    
    # Example 2: Section 2 (Classification) - Operator prediction
    run_experiment(
        data_path='data/section2/section2_mixed_1000.csv',
        model_type='densenet',
        section=2,
        epochs=100,
        batch_size=32,
        # DenseNet parameters
        hidden_dims=[256, 128, 64],
        dropout_rate=0.2,
        l2_reg=0.001
    )
    
    run_experiment(
        data_path='data/section2/section2_mixed_1000.csv',
        model_type='densenet',
        section=2,
        epochs=100,
        batch_size=32,
        # DenseNet parameters
        hidden_dims=[256, 128, 64],
        dropout_rate=0.2,
        l2_reg=0.001,
        use_feature_engineering=True  # Enable feature engineering
    )
    
    # # Example 3: Section 3 (Regression) - Custom formula
    # print("\nRunning Section 3 experiment...")
    # run_experiment(
    #     data_path='data/section3/section3_max_-_*.csv',
    #     model_type='densenet',
    #     section=3,
    #     epochs=75,
    #     batch_size=32,
    #     # DenseNet parameters
    #     hidden_dims=[256, 128, 64],
    #     dropout_rate=0.2,
    #     l2_reg=0.001
    # )

if __name__ == "__main__":
    main()