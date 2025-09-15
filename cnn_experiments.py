"""
Sign Language Digit Recognition using CNN
CPU-Optimized Implementation with Comprehensive Hyperparameter Analysis

This script implements a deep CNN for recognizing sign language digits (0-9)
with extensive hyperparameter tuning and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure TensorFlow for CPU optimization
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

class SignLanguageCNN:
    """
    A comprehensive CNN implementation for sign language digit recognition
    with extensive hyperparameter tuning capabilities.
    """
    
    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_train_categorical = None
        self.y_val_categorical = None
        self.y_test_categorical = None
        self.models = {}
        self.histories = {}
        self.results = []
        
    def load_and_explore_data(self):
        """Load the dataset and perform initial exploration."""
        print("Loading and exploring the dataset...")
        
        # Load the data
        X = np.load('data/X.npy')
        Y = np.load('data/Y.npy')
        
        print(f"Dataset shape: X={X.shape}, Y={Y.shape}")
        print(f"X data type: {X.dtype}, range: [{X.min()}, {X.max()}]")
        print(f"Y data type: {Y.dtype}")
        
        # Check if Y is one-hot encoded
        if Y.ndim == 2 and Y.shape[1] == 10:
            print("Labels are one-hot encoded")
            y_categorical = Y
            y_numeric = np.argmax(Y, axis=1)
        else:
            print("Labels are numeric")
            y_numeric = Y
            y_categorical = tf.keras.utils.to_categorical(Y, 10)
        
        # Print class distribution
        unique, counts = np.unique(y_numeric, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        # Visualize sample images
        self.visualize_samples(X, y_numeric)
        
        return X, y_categorical, y_numeric
    
    def visualize_samples(self, X, y_numeric, num_samples=20):
        """Visualize sample images from each class."""
        plt.figure(figsize=(15, 8))
        
        # Show 2 samples from each digit class
        for digit in range(10):
            digit_indices = np.where(y_numeric == digit)[0]
            for i in range(2):
                if i < len(digit_indices):
                    plt.subplot(4, 5, digit * 2 + i + 1)
                    plt.imshow(X[digit_indices[i]], cmap='gray')
                    plt.title(f'Digit {digit} (Sample {i+1})')
                    plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self, X, y_categorical, test_size=0.2, val_size=0.2):
        """Preprocess and split the data."""
        print("Preprocessing data...")
        
        # Normalize pixel values to [0, 1]
        X_normalized = X.astype('float32') / 255.0
        
        # Ensure proper shape for CNN (add channel dimension if needed)
        if len(X_normalized.shape) == 3:
            X_normalized = X_normalized.reshape(-1, 64, 64, 1)
        
        print(f"Normalized data shape: {X_normalized.shape}")
        print(f"Normalized data range: [{X_normalized.min()}, {X_normalized.max()}]")
        
        # Split into train and temp (which will be split into val and test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_normalized, y_categorical, test_size=test_size + val_size, 
            random_state=42, stratify=np.argmax(y_categorical, axis=1)
        )
        
        # Split temp into validation and test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio, 
            random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train_categorical = y_train
        self.y_val_categorical = y_val
        self.y_test_categorical = y_test
        self.y_train = np.argmax(y_train, axis=1)
        self.y_val = np.argmax(y_val, axis=1)
        self.y_test = np.argmax(y_test, axis=1)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_cnn_model(self, dropout_rate=0.3, l1_reg=0.0, l2_reg=0.001):
        """
        Create a CNN model with specified hyperparameters.
        
        Args:
            dropout_rate: Dropout rate for regularization
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
        """
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(64, 64, 1),
                         kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.Dropout(dropout_rate),
            
            # Output Layer
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def train_model(self, model, learning_rate=0.001, batch_size=32, epochs=100, 
                   patience=15, experiment_name="baseline"):
        """
        Train the CNN model with specified hyperparameters.
        
        Args:
            model: Compiled Keras model
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Early stopping patience
            experiment_name: Name for this experiment
        """
        print(f"Training model: {experiment_name}")
        print(f"Hyperparameters: LR={learning_rate}, Batch={batch_size}, "
              f"Epochs={epochs}, Patience={patience}")
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train_categorical,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val_categorical),
            callbacks=callbacks,
            verbose=1
        )
        
        # Store results
        self.models[experiment_name] = model
        self.histories[experiment_name] = history
        
        return history
    
    def evaluate_model(self, model, experiment_name="baseline"):
        """Comprehensive model evaluation with multiple metrics."""
        print(f"Evaluating model: {experiment_name}")
        
        # Predictions
        y_pred_proba = model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Basic metrics
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test_categorical, verbose=0)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification Report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # ROC-AUC for multiclass
        y_test_binary = label_binarize(self.y_test, classes=range(10))
        auc_scores = []
        for i in range(10):
            fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred_proba[:, i])
            auc_scores.append(auc(fpr, tpr))
        avg_auc = np.mean(auc_scores)
        
        # Store results
        result = {
            'experiment': experiment_name,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'auc_macro': avg_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'auc_per_class': auc_scores
        }
        
        self.results.append(result)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Macro Precision: {result['precision_macro']:.4f}")
        print(f"Macro Recall: {result['recall_macro']:.4f}")
        print(f"Macro F1-Score: {result['f1_macro']:.4f}")
        print(f"Macro AUC: {avg_auc:.4f}")
        
        return result
    
    def plot_training_history(self, experiment_names=None):
        """Plot training histories for comparison."""
        if experiment_names is None:
            experiment_names = list(self.histories.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for exp_name in experiment_names:
            history = self.histories[exp_name].history
            
            # Training & Validation Loss
            axes[0, 0].plot(history['loss'], label=f'{exp_name} - Train')
            axes[0, 0].plot(history['val_loss'], label=f'{exp_name} - Val')
            
            # Training & Validation Accuracy
            axes[0, 1].plot(history['accuracy'], label=f'{exp_name} - Train')
            axes[0, 1].plot(history['val_accuracy'], label=f'{exp_name} - Val')
        
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate (if available)
        if len(experiment_names) == 1:
            exp_name = experiment_names[0]
            history = self.histories[exp_name].history
            if 'lr' in history:
                axes[1, 0].plot(history['lr'])
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True)
        
        # Results comparison
        if len(self.results) > 0:
            exp_names = [r['experiment'] for r in self.results]
            accuracies = [r['test_accuracy'] for r in self.results]
            axes[1, 1].bar(exp_names, accuracies)
            axes[1, 1].set_title('Test Accuracy Comparison')
            axes[1, 1].set_xlabel('Experiment')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('training_histories.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, experiment_name="baseline"):
        """Plot confusion matrix for a specific experiment."""
        result = next((r for r in self.results if r['experiment'] == experiment_name), None)
        if result is None:
            print(f"No results found for experiment: {experiment_name}")
            return
        
        plt.figure(figsize=(10, 8))
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title(f'Confusion Matrix - {experiment_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'confusion_matrix_{experiment_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_baseline_experiment(self):
        """Run baseline experiment with default hyperparameters."""
        print("=" * 60)
        print("RUNNING BASELINE EXPERIMENT")
        print("=" * 60)
        
        # Create and train baseline model
        model = self.create_cnn_model(
            dropout_rate=0.3,
            l1_reg=0.0,
            l2_reg=0.001
        )
        
        history = self.train_model(
            model=model,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            patience=15,
            experiment_name="baseline"
        )
        
        # Evaluate model
        result = self.evaluate_model(model, "baseline")
        
        # Plot results
        self.plot_training_history(["baseline"])
        self.plot_confusion_matrix("baseline")
        
        return result


    def run_hyperparameter_experiments(self):
        """Run comprehensive hyperparameter tuning experiments."""
        print("=" * 60)
        print("RUNNING HYPERPARAMETER EXPERIMENTS")
        print("=" * 60)
        
        # Define hyperparameter grids
        experiments = [
            # Batch size experiments
            {
                'name': 'batch_16',
                'batch_size': 16,
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'epochs': 50,
                'patience': 10
            },
            {
                'name': 'batch_64',
                'batch_size': 64,
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'epochs': 50,
                'patience': 10
            },
            
            # Learning rate experiments
            {
                'name': 'lr_0005',
                'batch_size': 32,
                'learning_rate': 0.0005,
                'dropout_rate': 0.3,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'epochs': 50,
                'patience': 10
            },
            {
                'name': 'lr_002',
                'batch_size': 32,
                'learning_rate': 0.002,
                'dropout_rate': 0.3,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'epochs': 50,
                'patience': 10
            },
            
            # Dropout experiments
            {
                'name': 'dropout_02',
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.2,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'epochs': 50,
                'patience': 10
            },
            {
                'name': 'dropout_05',
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'epochs': 50,
                'patience': 10
            },
            
            # Regularization experiments
            {
                'name': 'l1_001',
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'l1_reg': 0.001,
                'l2_reg': 0.0,
                'epochs': 50,
                'patience': 10
            },
            {
                'name': 'l2_005',
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'l1_reg': 0.0,
                'l2_reg': 0.005,
                'epochs': 50,
                'patience': 10
            },
            
            # Early stopping patience experiments
            {
                'name': 'patience_5',
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'epochs': 50,
                'patience': 5
            },
            {
                'name': 'patience_20',
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'epochs': 50,
                'patience': 20
            }
        ]
        
        # Run each experiment
        for i, exp in enumerate(experiments):
            print(f"\n--- Experiment {i+1}/{len(experiments)}: {exp['name']} ---")
            
            # Create model with specific hyperparameters
            model = self.create_cnn_model(
                dropout_rate=exp['dropout_rate'],
                l1_reg=exp['l1_reg'],
                l2_reg=exp['l2_reg']
            )
            
            # Train model
            history = self.train_model(
                model=model,
                learning_rate=exp['learning_rate'],
                batch_size=exp['batch_size'],
                epochs=exp['epochs'],
                patience=exp['patience'],
                experiment_name=exp['name']
            )
            
            # Evaluate model
            result = self.evaluate_model(model, exp['name'])
            
            # Add hyperparameters to result
            result.update(exp)
            
            print(f"Experiment {exp['name']} completed!")
            print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        
        return experiments
    
    def create_results_summary(self):
        """Create a comprehensive summary of all experiments."""
        if not self.results:
            print("No results to summarize!")
            return
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("=" * 80)
        
        # Create DataFrame for easy analysis
        df_results = pd.DataFrame(self.results)
        
        # Sort by test accuracy
        df_results = df_results.sort_values('test_accuracy', ascending=False)
        
        # Display top performers
        print("\n🏆 TOP 5 PERFORMING MODELS:")
        print("-" * 50)
        top_5 = df_results.head(5)
        for idx, row in top_5.iterrows():
            print(f"{row['experiment']:12} | Acc: {row['test_accuracy']:.4f} | "
                  f"F1: {row['f1_macro']:.4f} | AUC: {row['auc_macro']:.4f}")
        
        # Hyperparameter analysis
        print(f"\n📊 HYPERPARAMETER ANALYSIS:")
        print("-" * 50)
        
        # Best hyperparameters
        best_model = df_results.iloc[0]
        print(f"Best Model: {best_model['experiment']}")
        print(f"  Batch Size: {best_model.get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {best_model.get('learning_rate', 'N/A')}")
        print(f"  Dropout Rate: {best_model.get('dropout_rate', 'N/A')}")
        print(f"  L1 Regularization: {best_model.get('l1_reg', 'N/A')}")
        print(f"  L2 Regularization: {best_model.get('l2_reg', 'N/A')}")
        print(f"  Patience: {best_model.get('patience', 'N/A')}")
        
        # Performance statistics
        print(f"\n📈 PERFORMANCE STATISTICS:")
        print("-" * 50)
        print(f"Mean Accuracy: {df_results['test_accuracy'].mean():.4f} ± {df_results['test_accuracy'].std():.4f}")
        print(f"Mean F1-Score: {df_results['f1_macro'].mean():.4f} ± {df_results['f1_macro'].std():.4f}")
        print(f"Mean AUC: {df_results['auc_macro'].mean():.4f} ± {df_results['auc_macro'].std():.4f}")
        print(f"Best Accuracy: {df_results['test_accuracy'].max():.4f}")
        print(f"Worst Accuracy: {df_results['test_accuracy'].min():.4f}")
        
        return df_results
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive visualizations of all experiments."""
        if not self.results:
            print("No results to plot!")
            return
        
        # Create DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall performance comparison
        ax1 = plt.subplot(3, 3, 1)
        experiments = df_results['experiment']
        accuracies = df_results['test_accuracy']
        bars = ax1.bar(range(len(experiments)), accuracies, color='skyblue', alpha=0.8)
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Test Accuracy by Experiment')
        ax1.set_xticks(range(len(experiments)))
        ax1.set_xticklabels(experiments, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Metrics comparison
        ax2 = plt.subplot(3, 3, 2)
        metrics = ['test_accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        x_pos = np.arange(len(experiments))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax2.bar(x_pos + i*width, df_results[metric], width, 
                   label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Score')
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x_pos + width * 1.5)
        ax2.set_xticklabels(experiments, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Training histories comparison (loss)
        ax3 = plt.subplot(3, 3, 3)
        for exp_name in self.histories.keys():
            history = self.histories[exp_name].history
            ax3.plot(history['val_loss'], label=exp_name, alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Loss')
        ax3.set_title('Validation Loss Curves')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Hyperparameter impact analysis
        # Batch size impact
        if 'batch_size' in df_results.columns:
            ax4 = plt.subplot(3, 3, 4)
            batch_experiments = df_results[df_results['batch_size'].notna()]
            if not batch_experiments.empty:
                ax4.scatter(batch_experiments['batch_size'], batch_experiments['test_accuracy'], 
                           s=100, alpha=0.7, color='red')
                ax4.set_xlabel('Batch Size')
                ax4.set_ylabel('Test Accuracy')
                ax4.set_title('Batch Size vs Accuracy')
                ax4.grid(True, alpha=0.3)
        
        # Learning rate impact
        if 'learning_rate' in df_results.columns:
            ax5 = plt.subplot(3, 3, 5)
            lr_experiments = df_results[df_results['learning_rate'].notna()]
            if not lr_experiments.empty:
                ax5.scatter(lr_experiments['learning_rate'], lr_experiments['test_accuracy'], 
                           s=100, alpha=0.7, color='green')
                ax5.set_xlabel('Learning Rate')
                ax5.set_ylabel('Test Accuracy')
                ax5.set_title('Learning Rate vs Accuracy')
                ax5.set_xscale('log')
                ax5.grid(True, alpha=0.3)
        
        # Dropout impact
        if 'dropout_rate' in df_results.columns:
            ax6 = plt.subplot(3, 3, 6)
            dropout_experiments = df_results[df_results['dropout_rate'].notna()]
            if not dropout_experiments.empty:
                ax6.scatter(dropout_experiments['dropout_rate'], dropout_experiments['test_accuracy'], 
                           s=100, alpha=0.7, color='purple')
                ax6.set_xlabel('Dropout Rate')
                ax6.set_ylabel('Test Accuracy')
                ax6.set_title('Dropout Rate vs Accuracy')
                ax6.grid(True, alpha=0.3)
        
        # 7. Confusion matrix for best model
        ax7 = plt.subplot(3, 3, 7)
        best_model = df_results.loc[df_results['test_accuracy'].idxmax()]
        cm = best_model['confusion_matrix']
        im = ax7.imshow(cm, interpolation='nearest', cmap='Blues')
        ax7.set_title(f'Best Model Confusion Matrix\n({best_model["experiment"]})')
        ax7.set_xlabel('Predicted')
        ax7.set_ylabel('Actual')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax7.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # 8. AUC comparison
        ax8 = plt.subplot(3, 3, 8)
        ax8.bar(range(len(experiments)), df_results['auc_macro'], color='orange', alpha=0.8)
        ax8.set_xlabel('Experiment')
        ax8.set_ylabel('AUC Score')
        ax8.set_title('AUC Scores by Experiment')
        ax8.set_xticks(range(len(experiments)))
        ax8.set_xticklabels(experiments, rotation=45, ha='right')
        ax8.grid(True, alpha=0.3)
        
        # 9. Training vs Validation accuracy for best model
        ax9 = plt.subplot(3, 3, 9)
        best_exp = best_model['experiment']
        if best_exp in self.histories:
            history = self.histories[best_exp].history
            ax9.plot(history['accuracy'], label='Training Accuracy', alpha=0.8)
            ax9.plot(history['val_accuracy'], label='Validation Accuracy', alpha=0.8)
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Accuracy')
            ax9.set_title(f'Best Model Training History\n({best_exp})')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the sign language CNN experiments."""
    print("Sign Language Digit Recognition - CPU Optimized CNN")
    print("=" * 60)
    
    # Initialize the CNN class
    cnn = SignLanguageCNN()
    
    # Load and explore data
    X, y_categorical, y_numeric = cnn.load_and_explore_data()
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = cnn.preprocess_data(X, y_categorical)
    
    # Run baseline experiment
    baseline_result = cnn.run_baseline_experiment()
    
    print("\nBaseline experiment completed!")
    print(f"Baseline Test Accuracy: {baseline_result['test_accuracy']:.4f}")
    
    # Ask user if they want to run hyperparameter experiments
    print("\n" + "=" * 60)
    print("Ready to run comprehensive hyperparameter experiments!")
    print("This will test different combinations of:")
    print("- Batch sizes (16, 32, 64)")
    print("- Learning rates (0.0005, 0.001, 0.002)")
    print("- Dropout rates (0.2, 0.3, 0.5)")
    print("- Regularization (L1, L2)")
    print("- Early stopping patience (5, 15, 20)")
    print("=" * 60)
    
    run_experiments = input("Run hyperparameter experiments? (y/n): ").lower().strip()
    
    if run_experiments == 'y':
        # Run hyperparameter experiments
        experiments = cnn.run_hyperparameter_experiments()
        
        # Create comprehensive analysis
        df_results = cnn.create_results_summary()
        
        # Create visualizations
        cnn.plot_comprehensive_analysis()
        
        print("\n🎉 All experiments completed successfully!")
        print("Check the generated plots and summary for detailed analysis.")
    else:
        print("\nSkipping hyperparameter experiments. Baseline results are available.")


if __name__ == "__main__":
    main()