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
        
        print(f"Feature columns: {feature_cols}")
        print(f"Sample data:")
        print(df.head())
        print(f"Data types:")
        print(df.dtypes)
        
        # Check for non-numeric values in features
        for col in feature_cols:
            non_numeric = df[col].apply(lambda x: not isinstance(x, (int, float, np.integer, np.floating)))
            if non_numeric.any():
                print(f"Non-numeric values found in {col}:")
                print(df[non_numeric][col].unique())
                
        # Convert features to numeric, replacing non-numeric with NaN
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        X = df[feature_cols].values
        
        # Handle NaN values in features and target
        target_mask = ~df['target'].isna()
        
        # Check for NaN in features (now they should be numeric)
        if np.isnan(X).any():
            print("NaN values found in features after conversion")
            feature_mask = ~np.isnan(X).any(axis=1)
        else:
            feature_mask = np.ones(len(X), dtype=bool)
        
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
        print(f"Valid samples after cleaning: {len(X)}")
        
    else:
        raise ValueError(f"Invalid section: {section}")
    
    print(f"Task type: {task_type}")
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # Final check for any remaining NaN values
    if np.isnan(X).any():
        print("Warning: NaN values still found in features, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    return X, y, task_type, num_classes

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train/validation/test sets with stratification
    
    Args:
        X, y: features and targets
        test_size: test set proportion
        val_size: validation set proportion (from remaining data)
        random_state: random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"Class distribution before split:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")
    
    # First split: separate test set
    # Use stratify only if all classes have at least 2 samples
    min_class_count = np.min(class_counts)
    use_stratify = min_class_count >= 2
    
    print(f"Using stratified split: {use_stratify} (min class count: {min_class_count})")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if use_stratify else None
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    
    # Check if stratification is still possible after first split
    unique_temp, temp_counts = np.unique(y_temp, return_counts=True)
    min_temp_count = np.min(temp_counts)
    use_stratify_temp = min_temp_count >= 2
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
        stratify=y_temp if use_stratify_temp else None
    )
    
    print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Print class distribution for each split
    print("Train class distribution:")
    unique_train, train_counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique_train, train_counts):
        print(f"  Class {cls}: {count} samples")
    
    print("Test class distribution:")
    unique_test, test_counts = np.unique(y_test, return_counts=True)
    for cls, count in zip(unique_test, test_counts):
        print(f"  Class {cls}: {count} samples")
    
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
                  epochs=100, batch_size=32, use_feature_engineering=False, **model_params):
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
    
    # 4. Apply feature engineering if requested
    if use_feature_engineering and section == 2:
        print("\n🔧 APPLYING FEATURE ENGINEERING")
        print("-" * 40)
        from trainer.feature_engineering import apply_feature_engineering, get_feature_names
        
        print(f"Original features shape: {X_train.shape}")
        print(f"Original features: [operand1, operand2, result]")
        print(f"Sample original: {X_train[0]}")
        
        X_train = apply_feature_engineering(X_train, use_feature_engineering=True)
        X_val = apply_feature_engineering(X_val, use_feature_engineering=True)
        X_test = apply_feature_engineering(X_test, use_feature_engineering=True)
        
        print(f"\nEnhanced features shape: {X_train.shape}")
        print(f"Features expanded from 3 to {X_train.shape[1]} dimensions")
        print(f"New features include: differences, ratios, comparisons, mathematical functions")
        print(f"Sample enhanced (first 10): {X_train[0][:10]}")
        
        # Check for any problematic values
        print(f"\nFeature statistics:")
        print(f"  Min: {X_train.min():.3f}, Max: {X_train.max():.3f}")
        print(f"  Mean: {X_train.mean():.3f}, Std: {X_train.std():.3f}")
        print(f"  NaN count: {np.isnan(X_train).sum()}")
        print(f"  Inf count: {np.isinf(X_train).sum()}")
        
        # Show feature names
        feature_names = get_feature_names()
        print(f"\nFeature names: {feature_names[:10]}... (showing first 10)")
    
    # 5. Normalize features if requested
    if normalize:
        X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
    
    # 6. Determine output dimension
    if task_type == 'regression':
        output_dim = 1
    else:  # classification
        output_dim = num_classes
    
    # 7. Create model
    model = create_model(
        model_type=model_type,
        task_type=task_type,
        input_dim=X_train.shape[1],
        output_dim=output_dim,
        **model_params
    )
    
    # 8. Setup trainer
    trainer = BaseTrainer(model, experiment_manager, task_type)
    
    # 9. Save experiment configuration
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
        'use_feature_engineering': use_feature_engineering,
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
    
    # 10. Train model
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 11. Evaluate model
    trainer.evaluate_model(X_test, y_test)
    
    # 12. Save model
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