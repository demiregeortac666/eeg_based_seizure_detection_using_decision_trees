#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG Seizure Detection GUI
A graphical interface for analyzing EEG data and detecting seizures
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score


class EEGSeizureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Seizure Detection - Analysis Tool")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize variables
        self.df = None
        self.feature_cols = None
        self.trained_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for data loading options
        self.top_frame = ttk.LabelFrame(self.main_frame, text="Data Loading")
        self.top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add file loading widgets
        self.file_path_var = tk.StringVar()
        ttk.Label(self.top_frame, text="Data File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_entry = ttk.Entry(self.top_frame, textvariable=self.file_path_var, width=60)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        self.browse_button = ttk.Button(self.top_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.load_button = ttk.Button(self.top_frame, text="Load Data", command=self.load_data)
        self.load_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Configure top frame grid
        self.top_frame.columnconfigure(1, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.modeling_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data Overview")
        self.notebook.add(self.visualization_tab, text="Data Visualization")
        self.notebook.add(self.modeling_tab, text="Modeling")
        self.notebook.add(self.results_tab, text="Results")
        
        # Set up each tab
        self.setup_data_tab()
        self.setup_visualization_tab()
        self.setup_modeling_tab()
        self.setup_results_tab()
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Check if we have a default file path to load
        if os.path.exists("output/selected_features.csv"):
            self.file_path_var.set("output/selected_features.csv")
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            
    def load_data(self):
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file first.")
            return
        
        try:
            self.status_var.set(f"Loading data from {file_path}...")
            self.root.update_idletasks()
            
            self.df = pd.read_csv(file_path)
            
            # Extract feature columns (exclude label, patient, file columns)
            exclude_cols = ['label', 'patient', 'file']
            self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
            
            # Update data view
            self.update_data_view()
            
            # Update data summary
            self.update_data_summary()
            
            # Enable visualization and modeling tabs
            self.notebook.tab(1, state="normal")
            self.notebook.tab(2, state="normal")
            
            # Update status
            self.status_var.set(f"Loaded {len(self.df)} records with {len(self.feature_cols)} features.")
            
            # Switch to data tab
            self.notebook.select(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Data loading failed.")
    
    def setup_data_tab(self):
        # Create frame for data summary
        self.summary_frame = ttk.LabelFrame(self.data_tab, text="Data Summary")
        self.summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add summary widgets
        self.records_var = tk.StringVar(value="Records: ")
        self.features_var = tk.StringVar(value="Features: ")
        self.classes_var = tk.StringVar(value="Class Distribution: ")
        self.patients_var = tk.StringVar(value="Patients: ")
        
        ttk.Label(self.summary_frame, textvariable=self.records_var).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(self.summary_frame, textvariable=self.features_var).grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(self.summary_frame, textvariable=self.classes_var).grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(self.summary_frame, textvariable=self.patients_var).grid(row=3, column=0, padx=5, pady=2, sticky="w")
        
        # Create frame for data display
        self.data_frame = ttk.LabelFrame(self.data_tab, text="Data Preview")
        self.data_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add treeview for data display
        columns = ["index"] + ["placeholder"]
        self.data_tree = ttk.Treeview(self.data_frame, columns=columns, show="headings")
        self.data_tree.heading("index", text="#")
        self.data_tree.heading("placeholder", text="Data not loaded")
        self.data_tree.column("index", width=50, stretch=False)
        
        # Add scrollbars
        x_scroll = ttk.Scrollbar(self.data_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        y_scroll = ttk.Scrollbar(self.data_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        
        # Pack widgets
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Initially disable visualization and modeling tabs
        self.notebook.tab(1, state="disabled")
        self.notebook.tab(2, state="disabled")
    
    def setup_visualization_tab(self):
        # Create control frame
        self.viz_control_frame = ttk.LabelFrame(self.visualization_tab, text="Visualization Controls")
        self.viz_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add visualization type dropdown
        ttk.Label(self.viz_control_frame, text="Plot Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.viz_type_var = tk.StringVar(value="Feature Distribution")
        self.viz_type_combo = ttk.Combobox(self.viz_control_frame, textvariable=self.viz_type_var, 
                                           values=["Feature Distribution", "Feature Correlation", "Class Distribution", "Feature Importance"])
        self.viz_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.viz_type_combo.bind("<<ComboboxSelected>>", self.update_visualization)
        
        # Feature selection (for distribution plots)
        ttk.Label(self.viz_control_frame, text="Feature:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.selected_feature_var = tk.StringVar()
        self.feature_combo = ttk.Combobox(self.viz_control_frame, textvariable=self.selected_feature_var, width=30)
        self.feature_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.feature_combo.bind("<<ComboboxSelected>>", self.update_visualization)
        
        # Add plot button
        self.plot_button = ttk.Button(self.viz_control_frame, text="Generate Plot", command=self.update_visualization)
        self.plot_button.grid(row=0, column=4, padx=5, pady=5)
        
        # Configure grid
        self.viz_control_frame.columnconfigure(3, weight=1)
        
        # Create frame for the visualization
        self.viz_frame = ttk.LabelFrame(self.visualization_tab, text="Visualization")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a figure for plotting
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_modeling_tab(self):
        # Create model configuration frame
        self.model_config_frame = ttk.LabelFrame(self.modeling_tab, text="Model Configuration")
        self.model_config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Model type selection
        ttk.Label(self.model_config_frame, text="Model Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_type_var = tk.StringVar(value="Random Forest")
        self.model_type_combo = ttk.Combobox(self.model_config_frame, textvariable=self.model_type_var, 
                                            values=["Random Forest", "Support Vector Machine", "K-Nearest Neighbors"])
        self.model_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Test size
        ttk.Label(self.model_config_frame, text="Test Size:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_entry = ttk.Entry(self.model_config_frame, textvariable=self.test_size_var, width=5)
        test_size_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Random state
        ttk.Label(self.model_config_frame, text="Random State:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.random_state_var = tk.IntVar(value=42)
        random_state_entry = ttk.Entry(self.model_config_frame, textvariable=self.random_state_var, width=5)
        random_state_entry.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        
        # Normalize data
        self.normalize_var = tk.BooleanVar(value=True)
        normalize_check = ttk.Checkbutton(self.model_config_frame, text="Normalize Data", variable=self.normalize_var)
        normalize_check.grid(row=0, column=6, padx=5, pady=5, sticky="w")
        
        # Train model button
        self.train_button = ttk.Button(self.model_config_frame, text="Train Model", command=self.train_model)
        self.train_button.grid(row=0, column=7, padx=15, pady=5)
        
        # Configure grid
        self.model_config_frame.columnconfigure(7, weight=1)
        
        # Create frame for model training results
        self.model_results_frame = ttk.LabelFrame(self.modeling_tab, text="Training Results")
        self.model_results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create two sub-frames side by side
        self.model_metrics_frame = ttk.Frame(self.model_results_frame)
        self.model_metrics_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.model_plot_frame = ttk.Frame(self.model_results_frame)
        self.model_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add model metrics text
        self.metrics_text = tk.Text(self.model_metrics_frame, wrap=tk.WORD, width=40, height=20)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add model visualization
        self.model_fig = Figure(figsize=(6, 6), dpi=100)
        self.model_canvas = FigureCanvasTkAgg(self.model_fig, master=self.model_plot_frame)
        self.model_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_results_tab(self):
        # Create controls frame
        self.result_controls_frame = ttk.LabelFrame(self.results_tab, text="Result Controls")
        self.result_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add visualization selector
        ttk.Label(self.result_controls_frame, text="Visualization:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.result_viz_var = tk.StringVar(value="Confusion Matrix")
        self.result_viz_combo = ttk.Combobox(self.result_controls_frame, textvariable=self.result_viz_var, 
                                           values=["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance"])
        self.result_viz_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.result_viz_combo.bind("<<ComboboxSelected>>", self.update_results_viz)
        
        # Add update button
        self.update_results_button = ttk.Button(self.result_controls_frame, text="Update Visualization", command=self.update_results_viz)
        self.update_results_button.grid(row=0, column=2, padx=15, pady=5)
        
        # Configure grid
        self.result_controls_frame.columnconfigure(2, weight=1)
        
        # Create frame for the visualization
        self.result_viz_frame = ttk.LabelFrame(self.results_tab, text="Results Visualization")
        self.result_viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a figure for plotting
        self.result_fig = Figure(figsize=(10, 8), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=self.result_viz_frame)
        self.result_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_data_view(self):
        # Clear existing tree
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Configure columns
        columns = ["index"] + list(self.df.columns)
        self.data_tree.configure(columns=columns)
        
        # Set column headings
        self.data_tree.heading("index", text="#")
        for col in self.df.columns:
            self.data_tree.heading(col, text=col)
            
            # Adjust column width based on content
            if self.df[col].dtype == 'float64':
                self.data_tree.column(col, width=100, stretch=True)
            elif col in ['label', 'patient', 'file']:
                self.data_tree.column(col, width=80, stretch=True)
            else:
                self.data_tree.column(col, width=120, stretch=True)
        
        self.data_tree.column("index", width=50, stretch=False)
        
        # Add data (first 100 rows)
        for i, row in self.df.head(100).iterrows():
            values = [i] + row.tolist()
            self.data_tree.insert("", "end", values=values)
    
    def update_data_summary(self):
        # Update summary information
        self.records_var.set(f"Records: {len(self.df)}")
        self.features_var.set(f"Features: {len(self.feature_cols)}")
        
        # Class distribution
        if 'label' in self.df.columns:
            class_dist = self.df['label'].value_counts().to_dict()
            self.classes_var.set(f"Class Distribution: {class_dist}")
        
        # Patient information
        if 'patient' in self.df.columns:
            patient_count = self.df['patient'].nunique()
            patients = sorted(self.df['patient'].unique())
            self.patients_var.set(f"Patients: {patient_count} unique ({', '.join(patients[:5])}{'...' if len(patients) > 5 else ''})")
        
        # Update feature dropdown for visualization
        if self.feature_cols:
            self.feature_combo['values'] = self.feature_cols
            if len(self.feature_cols) > 0:
                self.selected_feature_var.set(self.feature_cols[0])
    
    def update_visualization(self, event=None):
        if self.df is None:
            return
        
        viz_type = self.viz_type_var.get()
        self.fig.clear()
        
        try:
            if viz_type == "Feature Distribution":
                selected_feature = self.selected_feature_var.get()
                if not selected_feature or selected_feature not in self.feature_cols:
                    return
                
                ax = self.fig.add_subplot(111)
                
                # Split by class
                if 'label' in self.df.columns:
                    for label_val in sorted(self.df['label'].unique()):
                        subset = self.df[self.df['label'] == label_val]
                        label_name = "Seizure" if label_val == 1 else "Non-Seizure"
                        ax.hist(subset[selected_feature], alpha=0.5, bins=30, 
                                label=f'{label_name} (n={len(subset)})')
                    ax.legend()
                else:
                    ax.hist(self.df[selected_feature], bins=30)
                
                ax.set_title(f'Distribution of {selected_feature}')
                ax.set_xlabel(selected_feature)
                ax.set_ylabel('Frequency')
            
            elif viz_type == "Feature Correlation":
                # Select a subset of features for better visualization
                n_features = min(20, len(self.feature_cols))
                selected_features = self.feature_cols[:n_features]
                
                # Calculate correlation matrix
                corr_matrix = self.df[selected_features].corr()
                
                ax = self.fig.add_subplot(111)
                im = ax.imshow(corr_matrix, cmap='coolwarm')
                
                # Add feature names
                ax.set_xticks(np.arange(len(selected_features)))
                ax.set_yticks(np.arange(len(selected_features)))
                ax.set_xticklabels(selected_features, rotation=90)
                ax.set_yticklabels(selected_features)
                
                # Add colorbar
                self.fig.colorbar(im)
                ax.set_title('Feature Correlation Matrix')
                
            elif viz_type == "Class Distribution":
                if 'label' not in self.df.columns:
                    messagebox.showerror("Error", "No 'label' column found in the data.")
                    return
                
                ax = self.fig.add_subplot(111)
                counts = self.df['label'].value_counts().sort_index()
                labels = ["Non-Seizure", "Seizure"]
                
                ax.bar(labels, counts.values)
                ax.set_title('Class Distribution')
                ax.set_ylabel('Count')
                
                # Add count labels on bars
                for i, count in enumerate(counts.values):
                    ax.text(i, count + 5, str(count), ha='center')
            
            elif viz_type == "Feature Importance":
                if self.trained_model is None:
                    messagebox.showinfo("Info", "Please train a model first to see feature importance.")
                    return
                
                if not hasattr(self.trained_model, 'feature_importances_'):
                    messagebox.showinfo("Info", "The trained model doesn't provide feature importance.")
                    return
                
                importances = self.trained_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot top 20 features
                top_n = min(20, len(self.feature_cols))
                ax = self.fig.add_subplot(111)
                ax.bar(range(top_n), importances[indices[:top_n]])
                ax.set_xticks(range(top_n))
                ax.set_xticklabels([self.feature_cols[i] for i in indices[:top_n]], rotation=90)
                ax.set_title(f'Top {top_n} Feature Importance')
                ax.set_ylabel('Importance')
            
            self.fig.tight_layout()
            self.canvas.draw()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")
    
    def train_model(self):
        if self.df is None or len(self.feature_cols) == 0:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        if 'label' not in self.df.columns:
            messagebox.showerror("Error", "No 'label' column found in the data.")
            return
        
        try:
            self.status_var.set("Training model...")
            self.root.update_idletasks()
            
            # Prepare data
            X = self.df[self.feature_cols].values
            y = self.df['label'].values
            
            # Split data
            test_size = self.test_size_var.get()
            random_state = self.random_state_var.get()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y)
            
            # Normalize if requested
            if self.normalize_var.get():
                self.scaler = StandardScaler()
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)
            
            # Create and train model
            model_type = self.model_type_var.get()
            
            if model_type == "Random Forest":
                self.trained_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            elif model_type == "Support Vector Machine":
                from sklearn.svm import SVC
                self.trained_model = SVC(probability=True, random_state=random_state)
            elif model_type == "K-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsClassifier
                self.trained_model = KNeighborsClassifier(n_neighbors=5)
            
            self.trained_model.fit(self.X_train, self.y_train)
            
            # Evaluate model
            y_pred = self.trained_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, target_names=['Non-Seizure', 'Seizure'])
            
            # Display metrics
            metrics_text = f"Model: {model_type}\n\n"
            metrics_text += f"Accuracy: {accuracy:.4f}\n"
            metrics_text += f"F1 Score: {f1:.4f}\n\n"
            metrics_text += "Classification Report:\n"
            metrics_text += report
            
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, metrics_text)
            
            # Plot confusion matrix
            self.plot_confusion_matrix()
            
            # Enable results tab
            self.notebook.tab(3, state="normal")
            
            # Update status
            self.status_var.set(f"Model training complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_var.set("Model training failed.")
    
    def plot_confusion_matrix(self):
        if self.trained_model is None or self.X_test is None or self.y_test is None:
            return
        
        y_pred = self.trained_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.model_fig.clear()
        ax = self.model_fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        self.model_fig.colorbar(im)
        
        # Set labels
        classes = ['Non-Seizure', 'Seizure']
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, f"{cm[i, j]}", 
                              ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        self.model_fig.tight_layout()
        self.model_canvas.draw()
    
    def update_results_viz(self, event=None):
        if self.trained_model is None or self.X_test is None or self.y_test is None:
            messagebox.showinfo("Info", "Please train a model first.")
            return
        
        viz_type = self.result_viz_var.get()
        self.result_fig.clear()
        
        try:
            if viz_type == "Confusion Matrix":
                y_pred = self.trained_model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)
                
                ax = self.result_fig.add_subplot(111)
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                self.result_fig.colorbar(im)
                
                classes = ['Non-Seizure', 'Seizure']
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                ax.set_title('Confusion Matrix')
                
                # Add text annotations
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        text = ax.text(j, i, f"{cm[i, j]}", 
                                      ha="center", va="center", 
                                      color="white" if cm[i, j] > cm.max() / 2 else "black")
            
            elif viz_type == "ROC Curve":
                if not hasattr(self.trained_model, "predict_proba"):
                    messagebox.showinfo("Info", "The trained model doesn't support probability predictions for ROC curve.")
                    return
                
                y_scores = self.trained_model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                ax = self.result_fig.add_subplot(111)
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC)')
                ax.legend(loc="lower right")
            
            elif viz_type == "Precision-Recall Curve":
                if not hasattr(self.trained_model, "predict_proba"):
                    messagebox.showinfo("Info", "The trained model doesn't support probability predictions for Precision-Recall curve.")
                    return
                
                y_scores = self.trained_model.predict_proba(self.X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(self.y_test, y_scores)
                
                ax = self.result_fig.add_subplot(111)
                ax.plot(recall, precision, color='blue', lw=2)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_ylim([0.0, 1.05])
                ax.set_xlim([0.0, 1.0])
                ax.set_title('Precision-Recall Curve')
            
            elif viz_type == "Feature Importance":
                if not hasattr(self.trained_model, 'feature_importances_'):
                    messagebox.showinfo("Info", "The trained model doesn't provide feature importance.")
                    return
                
                importances = self.trained_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot top 20 features
                top_n = min(20, len(self.feature_cols))
                
                ax = self.result_fig.add_subplot(111)
                ax.bar(range(top_n), importances[indices[:top_n]])
                ax.set_xticks(range(top_n))
                ax.set_xticklabels([self.feature_cols[i] for i in indices[:top_n]], rotation=90)
                ax.set_title(f'Top {top_n} Feature Importance')
                ax.set_ylabel('Importance')
            
            self.result_fig.tight_layout()
            self.result_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EEGSeizureGUI(root)
    root.mainloop() 