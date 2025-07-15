import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from model.tabnet import TabNet
from model.self_regression import SelfRegression
from model.densenet import DenseNet
from trainer.base_trainer import BaseTrainer
from trainer.experiment import ExperimentManager

def load_and_preprocess_data(data_path, section):
    """
    Load and preprocess data based on section type
    
    Args:
        data_path: path to CSV file
        section: section number (1, 2, or 3)
    
    Returns:
        X, y, task_type, num_classes (if classification)
    """
    print(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove index column if exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Determine task type and prepare data based on section
    if section == 1 or section == 3:
        # Regression task
        task_type = 'regression'
        feature_cols = [col for col in df.columns if col.startswith('operand')]
        X = df[feature_cols].values
        y = df['target'].values
        
        # Handle NaN values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        num_classes = None
        
    elif section == 2:
        # Classification task
        task_type = 'classification'
        feature_cols = [col for col in df.columns if col.startswith('operand') or col == 'result']
        X = df[feature_cols].values
        
        # Handle NaN values in features and target
        target_mask = ~df['target'].isna()
        feature_mask = ~np.isnan(X).any(axis=1)
        valid_mask = target_mask & feature_mask
        
        X = X[valid_mask]
        y_labels = df['target'].values[valid_mask]
        
        # Encode target labels
        le = LabelEncoder()
        y = le.fit_transform(y_labels)
        num_classes = len(np.unique(y))
        
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {le.classes_}")
        print(f"Class distribution: {np.bincount(y)}")
        
    else:
        raise ValueError(f"Invalid section: {section}")
    
    print(f"Task type: {task_type}")
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # Check for any remaining NaN values
    if np.isnan(X).any():
        print("Warning: NaN values found in features")
        X = np.nan_to_num(X, nan=0.0)
    
    return X, y, task_type, num_classes

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train/validation/test sets
    
    Args:
        X, y: features and targets
        test_size: test set proportion
        val_size: validation set proportion (from remaining data)
        random_state: random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 20 else None
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
        stratify=y_temp if len(np.unique(y_temp)) < 20 else None
    )
    
    print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_features(X_train, X_val, X_test):
    """
    Normalize features using StandardScaler
    
    Args:
        X_train, X_val, X_test: feature arrays
    
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features normalized using StandardScaler")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def create_model(model_type, task_type, input_dim, output_dim, **model_params):
    """
    Create and build model based on type
    
    Args:
        model_type: 'tabnet', 'self_regression', or 'densenet'
        task_type: 'regression' or 'classification'
        input_dim: number of input features
        output_dim: number of output units
        **model_params: model-specific parameters
    
    Returns:
        model: compiled tensorflow model
    """
    print(f"Creating {model_type} model for {task_type}")
    
    if model_type.lower() == 'tabnet':
        model_class = TabNet(task_type=task_type, **model_params)
    elif model_type.lower() == 'self_regression':
        model_class = SelfRegression(task_type=task_type, **model_params)
    elif model_type.lower() == 'densenet':
        model_class = DenseNet(task_type=task_type, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Build model
    model = model_class.build_model(input_dim, output_dim)
    
    print(f"Model created successfully")
    print(f"Model parameters: {model.count_params():,}")
    
    return model

def run_experiment(data_path, model_type, section, experiment_id=None, 
                  test_size=0.2, val_size=0.2, normalize=True, 
                  epochs=100, batch_size=32, **model_params):
    """
    Run complete experiment pipeline
    
    Args:
        data_path: path to data CSV file
        model_type: 'tabnet', 'self_regression', or 'densenet'
        section: section number (1, 2, or 3)
        experiment_id: optional experiment ID
        test_size: test set proportion
        val_size: validation set proportion
        normalize: whether to normalize features
        epochs: number of training epochs
        batch_size: training batch size
        **model_params: model-specific parameters
    
    Returns:
        experiment_path: path to experiment results
    """
    print("=" * 50)
    print(f"STARTING EXPERIMENT")
    print(f"Model: {model_type}")
    print(f"Section: {section}")
    print(f"Data: {data_path}")
    print("=" * 50)
    
    # 1. Setup experiment
    experiment_manager = ExperimentManager(section, experiment_id)
    
    # 2. Load and preprocess data
    X, y, task_type, num_classes = load_and_preprocess_data(data_path, section)
    
    # 3. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size
    )
    
    # 4. Normalize features if requested
    if normalize:
        X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
    
    # 5. Determine output dimension
    if task_type == 'regression':
        output_dim = 1
    else:  # classification
        output_dim = num_classes
    
    # 6. Create model
    model = create_model(
        model_type=model_type,
        task_type=task_type,
        input_dim=X_train.shape[1],
        output_dim=output_dim,
        **model_params
    )
    
    # 7. Setup trainer
    trainer = BaseTrainer(model, experiment_manager, task_type)
    
    # 8. Save experiment configuration
    config = {
        'data_path': data_path,
        'model_type': model_type,
        'section': section,
        'task_type': task_type,
        'input_dim': X_train.shape[1],
        'output_dim': output_dim,
        'num_classes': num_classes,
        'test_size': test_size,
        'val_size': val_size,
        'normalize': normalize,
        'epochs': epochs,
        'batch_size': batch_size,
        'model_params': model_params,
        'data_shape': {
            'train': X_train.shape,
            'val': X_val.shape,
            'test': X_test.shape
        }
    }
    experiment_manager.save_config(config)
    
    # 9. Train model
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 10. Evaluate model
    trainer.evaluate_model(X_test, y_test)
    
    # 11. Save model
    trainer.save_model()
    
    print("=" * 50)
    print(f"EXPERIMENT COMPLETED")
    print(f"Results saved to: {experiment_manager.get_experiment_path()}")
    print("=" * 50)
    
    return experiment_manager.get_experiment_path()

def run_multiple_experiments(data_paths, model_types, sections, **kwargs):
    """
    Run multiple experiments
    
    Args:
        data_paths: list of data paths
        model_types: list of model types
        sections: list of sections
        **kwargs: common parameters for all experiments
    
    Returns:
        results: list of experiment paths
    """
    results = []
    
    for data_path in data_paths:
        for model_type in model_types:
            for section in sections:
                try:
                    exp_path = run_experiment(
                        data_path=data_path,
                        model_type=model_type,
                        section=section,
                        **kwargs
                    )
                    results.append({
                        'data_path': data_path,
                        'model_type': model_type,
                        'section': section,
                        'experiment_path': exp_path,
                        'status': 'success'
                    })
                except Exception as e:
                    print(f"Error in experiment: {e}")
                    results.append({
                        'data_path': data_path,
                        'model_type': model_type,
                        'section': section,
                        'experiment_path': None,
                        'status': 'failed',
                        'error': str(e)
                    })
    
    return results