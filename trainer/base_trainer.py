import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf

class BaseTrainer:
    def __init__(self, model, experiment_manager, task_type='regression'):
        """
        Base trainer for all models
        
        Args:
            model: compiled tensorflow model
            experiment_manager: ExperimentManager instance
            task_type: 'regression' or 'classification'
        """
        self.model = model
        self.experiment_manager = experiment_manager
        self.task_type = task_type
        self.history = None
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              verbose=1, callbacks=None, **kwargs):
        """
        Train the model
        
        Args:
            X_train, y_train: training data
            X_val, y_val: validation data
            epochs: number of epochs
            batch_size: batch size
            verbose: verbosity level
            callbacks: list of keras callbacks
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        
        # Check for NaN values
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print("Warning: NaN values detected in training data")
            X_train = np.nan_to_num(X_train, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
        
        if np.isnan(X_val).any() or np.isnan(y_val).any():
            print("Warning: NaN values detected in validation data")
            X_val = np.nan_to_num(X_val, nan=0.0)
            y_val = np.nan_to_num(y_val, nan=0.0)
        
        # Default callbacks with more conservative settings
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
                tf.keras.callbacks.TerminateOnNaN()
            ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs
        )
        
        print("Training completed!")
        
        # Save training logs
        self.save_training_logs(self.history)
        
        # Plot training curves
        self.plot_training_curves(self.history)
        
        return self.history
    
    def save_training_logs(self, history):
        """Save training logs to CSV and JSON"""
        logs_path = self.experiment_manager.get_logs_path()
        
        # Save as CSV
        df = pd.DataFrame(history.history)
        csv_path = os.path.join(logs_path, "training_log.csv")
        df.to_csv(csv_path, index_label='epoch')
        
        # Save as JSON
        json_path = os.path.join(logs_path, "training_history.json")
        with open(json_path, 'w') as f:
            # Convert numpy types to python types for JSON serialization
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        print(f"Training logs saved to {logs_path}")
    
    def plot_training_curves(self, history):
        """Plot and save training curves"""
        plots_path = self.experiment_manager.get_plots_path()
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Plot loss curves
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics plot
        plt.subplot(1, 3, 2)
        if self.task_type == 'regression':
            plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
            plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
            plt.title('Mean Absolute Error', fontsize=14, fontweight='bold')
            plt.ylabel('MAE')
        else:  # classification
            plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            plt.title('Model Accuracy', fontsize=14, fontweight='bold')
            plt.ylabel('Accuracy')
        
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], linewidth=2, color='red')
            plt.title('Learning Rate', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12)
            plt.title('Learning Rate', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_path, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model and save results"""
        print("Evaluating model...")
        
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        if self.task_type == 'classification':
            # Convert probabilities to class predictions
            y_pred_classes = np.argmax(y_pred, axis=1)
            self.save_classification_metrics(y_test, y_pred_classes)
            self.save_confusion_matrix(y_test, y_pred_classes)
        else:  # regression
            self.save_regression_metrics(y_test, y_pred)
        
        print("Model evaluation completed!")
        
        return y_pred
    
    def save_confusion_matrix(self, y_true, y_pred):
        """Save confusion matrix plot"""
        plots_path = self.experiment_manager.get_plots_path()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    square=True, linewidths=0.5)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Save plot
        plot_path = os.path.join(plots_path, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {plot_path}")
    
    def save_classification_metrics(self, y_true, y_pred):
        """Save classification performance metrics"""
        results_path = self.experiment_manager.get_results_path()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        # Classification report
        report = classification_report(y_true, y_pred)
        
        # Save detailed report
        report_path = os.path.join(results_path, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-Score (weighted): {f1:.4f}\n")
            f.write(f"Precision (weighted): {precision:.4f}\n")
            f.write(f"Recall (weighted): {recall:.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write("-" * 30 + "\n")
            f.write(report)
        
        # Save metrics as JSON
        metrics_dict = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        json_path = os.path.join(results_path, "classification_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"Classification metrics saved to {results_path}")
        print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    def save_regression_metrics(self, y_true, y_pred):
        """Save regression performance metrics"""
        results_path = self.experiment_manager.get_results_path()
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        # Save detailed report
        report_path = os.path.join(results_path, "regression_metrics.txt")
        with open(report_path, 'w') as f:
            f.write("REGRESSION PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"R² Score: {r2:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
            f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        
        # Save metrics as JSON
        metrics_dict = {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse)
        }
        
        json_path = os.path.join(results_path, "regression_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Create prediction vs actual plot
        self.plot_regression_results(y_true, y_pred)
        
        print(f"Regression metrics saved to {results_path}")
        print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    def plot_regression_results(self, y_true, y_pred):
        """Plot regression results"""
        plots_path = self.experiment_manager.get_plots_path()
        
        plt.figure(figsize=(12, 5))
        
        # Prediction vs Actual scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred.flatten()
        plt.scatter(y_pred, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_path, "regression_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Regression plots saved to {plot_path}")
    
    def save_model(self, model_name=None):
        """Save model weights and architecture"""
        if model_name is None:
            model_path = self.experiment_manager.get_model_save_path()
        else:
            model_path = f"model/saved_models/{model_name}"
        
        # Save weights
        self.model.save_weights(f"{model_path}.weights.h5")

        # Save architecture
        with open(f"{model_path}_architecture.json", 'w') as f:
            f.write(self.model.to_json())
        
        print(f"Model saved to {model_path}")
        
        return model_path