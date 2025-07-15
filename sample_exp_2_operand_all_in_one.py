#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Operator Classification using TabNet
- Feature Engineering vs Baseline comparison
- Comprehensive visualization and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TabNet:
    """TabNet implementation for operator classification"""
    
    def __init__(self, task_type='classification', n_d=32, n_a=32, n_steps=3, gamma=1.3):
        self.task_type = task_type
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.model = None

    def build_model(self, input_dim, output_dim):
        """Build TabNet model"""
        inputs = layers.Input(shape=(input_dim,), name="input")
        
        # Initial batch norm
        x = layers.BatchNormalization(name="initial_bn")(inputs)
        
        # Initialize prior - same shape as input features
        prior = layers.Lambda(lambda x: tf.ones_like(x), name="prior_init")(inputs)
        decision_outputs = []
        
        for step in range(self.n_steps):
            # Feature selection mask
            mask = self.attentive_transformer(
                inputs, prior, name_prefix=f"step_{step}_attention"
            )
            
            # Apply mask to original input features
            masked_features = layers.Multiply(name=f"step_{step}_mask_apply")([inputs, mask])
            
            # Feature transformation
            transformed = layers.Dense(self.n_d + self.n_a, activation='relu',
                                     name=f"step_{step}_transform")(masked_features)
            transformed = layers.BatchNormalization(name=f"step_{step}_bn")(transformed)
            
            # Decision output
            decision_out = layers.Dense(self.n_d, activation='relu',
                                      name=f"step_{step}_decision")(transformed)
            decision_outputs.append(decision_out)
            
            # Update prior for next step
            prior = layers.Lambda(
                lambda inputs: inputs[0] * (self.gamma - inputs[1]),
                output_shape=lambda input_shape: input_shape[0],
                name=f"step_{step}_prior_update"
            )([prior, mask])
        
        # Aggregate decision outputs
        if len(decision_outputs) == 1:
            decision_sum = decision_outputs[0]
        else:
            decision_sum = layers.Add(name="decision_aggregation")(decision_outputs)
        
        # Final output layer
        outputs = layers.Dense(output_dim, activation='softmax', name="output")(decision_sum)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="TabNet")
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

    def attentive_transformer(self, features, prior, name_prefix=""):
        """Attentive transformer for feature selection"""
        # Project to attention space
        attention = layers.Dense(self.n_a, name=f"{name_prefix}_dense")(features)
        attention = layers.BatchNormalization(name=f"{name_prefix}_bn")(attention)
        
        # Project back to feature space for mask
        mask_logits = layers.Dense(features.shape[-1], name=f"{name_prefix}_mask_proj")(attention)
        
        # Apply prior
        mask_logits = layers.Multiply(name=f"{name_prefix}_prior_apply")([mask_logits, prior])
        
        # Softmax to get attention mask
        mask = layers.Activation('softmax', name=f"{name_prefix}_softmax")(mask_logits)
        
        return mask

def generate_operator_data(num_samples=1000):
    """Generate synthetic operator data"""
    operators = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'truediv': lambda x, y: x / (y + 1e-8),
        'max': lambda x, y: np.maximum(x, y),
        'min': lambda x, y: np.minimum(x, y),
        'mod': lambda x, y: x % (y + 1e-8),
        'pow': lambda x, y: np.power(x, np.clip(y, 0, 3)),  # Limited power
        'abssub': lambda x, y: np.abs(x - y),
        'hypot': lambda x, y: np.sqrt(x**2 + y**2)
    }
    
    data = []
    labels = []
    
    for op_name, op_func in operators.items():
        for _ in range(num_samples):
            # Generate random operands
            operand1 = np.random.uniform(-10, 10)
            operand2 = np.random.uniform(-10, 10)
            
            # Calculate result
            try:
                result = op_func(operand1, operand2)
                if np.isfinite(result):
                    data.append([operand1, operand2, result])
                    labels.append(op_name)
            except:
                continue
    
    return np.array(data), np.array(labels)

def extract_safe_features(operand1, operand2, result):
    """Extract safe features without data leakage"""
    op1, op2, res = operand1, operand2, result
    eps = 1e-8
    
    features = [op1, op2, res]
    
    # Safe differences and ratios
    features.extend([
        res - op1, res - op2, op1 - op2,
        np.abs(res - op1), np.abs(res - op2), np.abs(op1 - op2),
        res / (op1 + eps), res / (op2 + eps), op1 / (op2 + eps), op2 / (op1 + eps)
    ])
    
    # Safe comparisons
    features.extend([
        (res > op1).astype(float), (res > op2).astype(float),
        (res < op1).astype(float), (res < op2).astype(float),
        (res > 0).astype(float), (op1 > op2).astype(float)
    ])
    
    # Magnitude and signs
    features.extend([
        np.abs(res), np.abs(op1), np.abs(op2),
        np.sign(res), np.sign(op1), np.sign(op2)
    ])
    
    # Logarithmic (safe)
    features.extend([
        np.log(np.abs(res) + eps), np.log(np.abs(op1) + eps), np.log(np.abs(op2) + eps)
    ])
    
    return np.column_stack(features)

def get_feature_info(use_feature_engineering=False):
    """Get detailed feature information"""
    if use_feature_engineering:
        feature_names = [
            'operand1', 'operand2', 'result',
            'res_minus_op1', 'res_minus_op2', 'op1_minus_op2',
            'abs_res_minus_op1', 'abs_res_minus_op2', 'abs_op1_minus_op2',
            'res_div_op1', 'res_div_op2', 'op1_div_op2', 'op2_div_op1',
            'res_gt_op1', 'res_gt_op2', 'res_lt_op1', 'res_lt_op2',
            'res_positive', 'op1_gt_op2',
            'abs_res', 'abs_op1', 'abs_op2',
            'sign_res', 'sign_op1', 'sign_op2',
            'log_abs_res', 'log_abs_op1', 'log_abs_op2'
        ]
        feature_types = [
            'Basic', 'Basic', 'Basic',
            'Difference', 'Difference', 'Difference', 
            'Abs_Difference', 'Abs_Difference', 'Abs_Difference',
            'Ratio', 'Ratio', 'Ratio', 'Ratio',
            'Comparison', 'Comparison', 'Comparison', 'Comparison', 'Comparison', 'Comparison',
            'Magnitude', 'Magnitude', 'Magnitude',
            'Sign', 'Sign', 'Sign',
            'Logarithmic', 'Logarithmic', 'Logarithmic'
        ]
    else:
        feature_names = ['operand1', 'operand2', 'result']
        feature_types = ['Basic', 'Basic', 'Basic']
    
    return feature_names, feature_types

def prepare_data(use_feature_engineering=False, X_raw=None, y_raw=None):
    """Prepare dataset for training"""
    print(f"🔄 Generating data with feature engineering: {use_feature_engineering}")
    
    # Generate data only once, or use provided data
    if X_raw is None or y_raw is None:
        X, y = generate_operator_data(num_samples=1500)
    else:
        X, y = X_raw, y_raw
    
    # Get operator information (only print for baseline)
    if not use_feature_engineering:
        operators = ['add', 'sub', 'mul', 'truediv', 'max', 'min', 'mod', 'pow', 'abssub', 'hypot']
        unique_operators, operator_counts = np.unique(y, return_counts=True)
        
        print(f"\n📋 OPERATOR CLASSES INFORMATION")
        print("-" * 40)
        print(f"Total number of classes: {len(unique_operators)}")
        print(f"Class names: {list(unique_operators)}")
        print("Class distribution:")
        for op, count in zip(unique_operators, operator_counts):
            print(f"  {op:>8}: {count:>4} samples")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Feature information
    feature_names, feature_types = get_feature_info(use_feature_engineering)
    
    print(f"\n📊 FEATURE INFORMATION")
    print("-" * 40)
    
    # Apply feature engineering if requested
    if use_feature_engineering:
        operand1, operand2, result = X[:, 0], X[:, 1], X[:, 2]
        X = extract_safe_features(operand1, operand2, result)
        print(f"✅ Features expanded from 3 to {X.shape[1]} dimensions")
        
        # Group features by type
        feature_type_counts = {}
        for ftype in feature_types:
            feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1
        
        print("Feature types breakdown:")
        for ftype, count in feature_type_counts.items():
            print(f"  {ftype:>15}: {count:>2} features")
        
        print("\nAll feature names:")
        for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
            print(f"  [{i:2d}] {name:>20} ({ftype})")
            
    else:
        print(f"📊 Using basic features: {X.shape[1]} dimensions")
        print("Feature names:")
        for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
            print(f"  [{i:2d}] {name:>20} ({ftype})")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"\n📦 DATA SPLIT INFORMATION")
    print("-" * 40)
    print(f"Training set:   {X_train.shape[0]:>5} samples × {X_train.shape[1]:>2} features")
    print(f"Validation set: {X_val.shape[0]:>5} samples × {X_val.shape[1]:>2} features")
    print(f"Test set:       {X_test.shape[0]:>5} samples × {X_test.shape[1]:>2} features")
    
    # Class distribution in splits (only print for baseline)
    if not use_feature_engineering:
        unique_operators, _ = np.unique(y, return_counts=True)
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        
        print(f"\nClass distribution in train/test:")
        print("Class     Train  Test")
        print("-" * 20)
        for i, op in enumerate(unique_operators):
            train_count = train_counts[train_classes == le.transform([op])[0]][0] if op in le.inverse_transform(train_classes) else 0
            test_count = test_counts[test_classes == le.transform([op])[0]][0] if op in le.inverse_transform(test_classes) else 0
            print(f"{op:>8}: {train_count:>4}   {test_count:>4}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test), le, scaler, (X, y)

def train_model(data, model_name):
    """Train TabNet model"""
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    print(f"\n🚀 Training {model_name}")
    print("-" * 50)
    
    # Build model
    tabnet = TabNet(n_d=32, n_a=32, n_steps=3)
    model = tabnet.build_model(X_train.shape[1], len(np.unique(y_train)))
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_loss': test_loss
    }
    
    print(f"✅ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return model, history, y_pred_classes, metrics

def plot_training_curves(histories, model_names):
    """Plot training curves comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    
    for i, (history, name) in enumerate(zip(histories, model_names)):
        color = ['blue', 'red'][i]
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label=f'{name} Train', color=color, linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label=f'{name} Val', color=color, linestyle='--', linewidth=2)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label=f'{name} Train', color=color, linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label=f'{name} Val', color=color, linestyle='--', linewidth=2)
    
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
    
    # Metrics comparison
    metrics_comparison = pd.DataFrame({
        'Baseline': [histories[0].history['val_accuracy'][-1], histories[0].history['val_loss'][-1]],
        'Feature Eng': [histories[1].history['val_accuracy'][-1], histories[1].history['val_loss'][-1]]
    }, index=['Val Accuracy', 'Val Loss'])
    
    axes[1, 0].bar(metrics_comparison.columns, metrics_comparison.loc['Val Accuracy'], 
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 0].set_title('Final Validation Accuracy', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    
    axes[1, 1].bar(metrics_comparison.columns, metrics_comparison.loc['Val Loss'], 
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title('Final Validation Loss', fontweight='bold')
    axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_fe, class_names):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline confusion matrix
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Baseline Model\nConfusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Feature engineering confusion matrix
    cm_fe = confusion_matrix(y_test, y_pred_fe)
    sns.heatmap(cm_fe, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Feature Engineering Model\nConfusion Matrix', fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(metrics_baseline, metrics_fe):
    """Plot metrics comparison"""
    metrics_df = pd.DataFrame({
        'Baseline': [metrics_baseline['accuracy'], metrics_baseline['precision'], 
                    metrics_baseline['recall'], metrics_baseline['f1']],
        'Feature Engineering': [metrics_fe['accuracy'], metrics_fe['precision'], 
                               metrics_fe['recall'], metrics_fe['f1']]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax, color=['blue', 'red'], alpha=0.7, width=0.8)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(title='Model Type')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(metrics_df.iterrows()):
        for j, value in enumerate(row):
            ax.text(i + (j-0.5)*0.4, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_tsne_visualization(X_test, y_test, y_pred_baseline, y_pred_fe, class_names):
    """Plot t-SNE visualization"""
    print("🔄 Computing t-SNE embeddings...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True labels
    scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, 
                             cmap='tab10', alpha=0.7, s=20)
    axes[0].set_title('True Labels', fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Baseline predictions
    scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_baseline, 
                             cmap='tab10', alpha=0.7, s=20)
    axes[1].set_title('Baseline Predictions', fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    # Feature engineering predictions
    scatter = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_fe, 
                             cmap='tab10', alpha=0.7, s=20)
    axes[2].set_title('Feature Engineering Predictions', fontweight='bold')
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal', 
                       pad=0.1, aspect=30, shrink=0.8)
    cbar.set_label('Operator Class')
    
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_classification_reports(y_test, y_pred_baseline, y_pred_fe, class_names):
    """Print detailed classification reports"""
    print("\n" + "="*80)
    print("📊 DETAILED CLASSIFICATION REPORTS")
    print("="*80)
    
    print("\n🔵 BASELINE MODEL:")
    print("-" * 40)
    print(classification_report(y_test, y_pred_baseline, target_names=class_names))
    
    print("\n🔴 FEATURE ENGINEERING MODEL:")
    print("-" * 40)
    print(classification_report(y_test, y_pred_fe, target_names=class_names))

def main():
    """Main execution function"""
    print("🎯 OPERATOR CLASSIFICATION WITH TABNET")
    print("="*60)
    print("Comparing Baseline vs Feature Engineering approaches")
    print("="*60)
    
    # Show experiment overview
    print("\n🧪 EXPERIMENT OVERVIEW")
    print("-" * 40)
    print("📌 Objective: Compare TabNet performance with/without feature engineering")
    print("📌 Task: Multi-class operator classification")
    print("📌 Model: TabNet with attention mechanism")
    print("📌 Evaluation: Accuracy, Precision, Recall, F1-Score + Visualizations")
    
    # Generate data once
    print("\n" + "="*60)
    print("📊 DATASET PREPARATION")
    print("="*60)
    
    print("\n1️⃣ BASELINE DATASET (No Feature Engineering)")
    print("="*50)
    # Baseline data (3 features)
    data_baseline, le, scaler_baseline, (X_raw, y_raw) = prepare_data(use_feature_engineering=False)
    
    print("\n2️⃣ ENHANCED DATASET (With Feature Engineering)")
    print("="*50)
    # Feature engineering data (same raw data, different features)
    data_fe, _, scaler_fe, _ = prepare_data(use_feature_engineering=True, X_raw=X_raw, y_raw=y_raw)
    
    # Verify test sets are the same size
    print(f"\n🔍 VERIFICATION:")
    print(f"Baseline test set size: {len(data_baseline[5])}")
    print(f"Feature Eng test set size: {len(data_fe[5])}")
    assert len(data_baseline[5]) == len(data_fe[5]), "Test sets must have same size!"
    print("✅ Test sets have consistent size")
    
    # Train models
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING")
    print("="*60)
    
    model_baseline, history_baseline, y_pred_baseline, metrics_baseline = train_model(
        data_baseline, "Baseline TabNet"
    )
    
    model_fe, history_fe, y_pred_fe, metrics_fe = train_model(
        data_fe, "Feature Engineering TabNet"
    )
    
    # Get test data and class names
    y_test = data_baseline[5]  # Same test labels for both
    class_names = le.classes_
    
    print(f"\n🔍 FINAL VERIFICATION:")
    print(f"Test labels size: {len(y_test)}")
    print(f"Baseline predictions size: {len(y_pred_baseline)}")
    print(f"Feature Eng predictions size: {len(y_pred_fe)}")
    
    # Visualizations
    print("\n" + "="*60)
    print("📈 GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Training curves
    print("📊 1. Training curves comparison...")
    plot_training_curves([history_baseline, history_fe], ['Baseline', 'Feature Engineering'])
    
    # 2. Confusion matrices
    print("📊 2. Confusion matrices...")
    plot_confusion_matrices(y_test, y_pred_baseline, y_pred_fe, class_names)
    
    # 3. Metrics comparison
    print("📊 3. Metrics comparison...")
    plot_metrics_comparison(metrics_baseline, metrics_fe)
    
    # 4. t-SNE visualization (use feature engineered test data for visualization)
    print("📊 4. t-SNE visualization...")
    plot_tsne_visualization(data_fe[2], y_test, y_pred_baseline, y_pred_fe, class_names)
    
    # 5. Classification reports
    print_classification_reports(y_test, y_pred_baseline, y_pred_fe, class_names)
    
    # Summary
    print("\n" + "="*80)
    print("🎯 FINAL SUMMARY")
    print("="*80)
    print(f"Baseline Model (3 features):")
    print(f"  Accuracy: {metrics_baseline['accuracy']:.4f}")
    print(f"  F1-Score: {metrics_baseline['f1']:.4f}")
    print(f"  Precision: {metrics_baseline['precision']:.4f}")
    print(f"  Recall: {metrics_baseline['recall']:.4f}")
    
    print(f"\nFeature Engineering Model (26 features):")
    print(f"  Accuracy: {metrics_fe['accuracy']:.4f}")
    print(f"  F1-Score: {metrics_fe['f1']:.4f}")
    print(f"  Precision: {metrics_fe['precision']:.4f}")
    print(f"  Recall: {metrics_fe['recall']:.4f}")
    
    improvement = (metrics_fe['accuracy'] - metrics_baseline['accuracy']) * 100
    print(f"\n🚀 Improvement: {improvement:+.2f}% accuracy gain with feature engineering")
    
    print("\n📁 Generated files:")
    print("  - training_curves_comparison.png")
    print("  - confusion_matrices_comparison.png") 
    print("  - metrics_comparison.png")
    print("  - tsne_visualization.png")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()