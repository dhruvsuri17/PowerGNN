# Power Grid Forecasting with Global XGBoost

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# For XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
    print("XGBoost successfully imported")
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not available. Install with: pip install xgboost")

# Add parent directory to path to find src module if needed
sys.path.append('../')
sys.path.append('../scripts')

class NaiveModel:
    """Persistence forecast model"""
    def __init__(self, forecast_horizon):
        self.forecast_horizon = forecast_horizon
    
    def predict(self, x):
        # Use the last observed voltage value for all future timesteps
        last_values = x['voltage_raw'][-1]
        return np.tile(last_values, (self.forecast_horizon, 1))

class MeanModel:
    """Mean forecast model"""
    def __init__(self, forecast_horizon):
        self.forecast_horizon = forecast_horizon
    
    def predict(self, x):
        # Use the mean of historical values for all future timesteps
        mean_values = np.mean(x['voltage_raw'], axis=0)
        return np.tile(mean_values, (self.forecast_horizon, 1))

class GlobalXGBoostModel:
    """
    A global XGBoost model that forecasts all nodes and horizons together.
    Instead of creating separate models for each node and horizon, this approach:
    1. Creates a single XGBoost model
    2. Uses node ID and horizon as features
    3. Leverages patterns across different nodes and horizons
    """
    def __init__(self, n_nodes, forecast_horizon, model=None):
        self.n_nodes = n_nodes
        self.forecast_horizon = forecast_horizon
        self.model = model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train a global XGBoost model on the entire dataset
        
        Args:
            X_train: List of input sequences
            y_train: List of target values
            X_val: Validation inputs
            y_val: Validation targets
        """
        if not HAS_XGBOOST:
            print("XGBoost not available. Install with: pip install xgboost")
            return
        
        print("Preparing training data for global XGBoost model...")
        
        # Create training samples for all nodes and horizons
        X_samples = []
        y_samples = []
        
        # Process each training sequence
        for seq_idx, (x_seq, y_seq) in enumerate(zip(X_train, y_train)):
            # Extract features
            features_dict = self._extract_global_features(x_seq)
            
            # For each node and horizon, create a training sample
            for node_idx in range(self.n_nodes):
                for horizon in range(self.forecast_horizon):
                    # Create feature vector with node and horizon info
                    features = features_dict.copy()
                    features['node_idx'] = node_idx
                    features['horizon'] = horizon
                    X_samples.append(features)
                    
                    # Target is the voltage at this node and horizon
                    y_samples.append(y_seq[horizon, node_idx])
        
        # Convert to DataFrame for easier handling
        X_df = pd.DataFrame(X_samples)
        y_array = np.array(y_samples)
        
        print(f"Created {len(X_df)} training samples")
        
        # Train XGBoost model
        print("Training global XGBoost model...")
        params = {
            'objective': 'reg:squarederror',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0,
            'alpha': 0.1,
            'lambda': 1,
            'nthread': -1,
            'verbosity': 1
        }
        
        # Create validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_samples = []
            y_val_samples = []
            
            for seq_idx, (x_seq, y_seq) in enumerate(zip(X_val, y_val)):
                features_dict = self._extract_global_features(x_seq)
                
                for node_idx in range(self.n_nodes):
                    for horizon in range(self.forecast_horizon):
                        features = features_dict.copy()
                        features['node_idx'] = node_idx
                        features['horizon'] = horizon
                        X_val_samples.append(features)
                        y_val_samples.append(y_seq[horizon, node_idx])
            
            X_val_df = pd.DataFrame(X_val_samples)
            y_val_array = np.array(y_val_samples)
            
            eval_set = [(X_df, y_array), (X_val_df, y_val_array)]
        
        # Train model
        dtrain = xgb.DMatrix(X_df, label=y_array)
        
        if eval_set:
            dval = xgb.DMatrix(X_val_df, label=y_val_array)
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=200,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=20,
                verbose_eval=10
            )
        else:
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=200,
                verbose_eval=10
            )
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
        print("\nFeature Importance:")
        for feat, score in list(importance.items())[:10]:  # Top 10 features
            print(f"  {feat}: {score}")
    
    def predict(self, X_test):
        """
        Make predictions for all nodes and horizons
        
        Args:
            X_test: List of test sequences
            
        Returns:
            Array of predictions: shape [n_samples, forecast_horizon, n_nodes]
        """
        if self.model is None:
            print("Model not trained yet")
            # Return default predictions (last known value)
            return np.array([np.tile(x['voltage_raw'][-1], (self.forecast_horizon, 1)) for x in X_test])
        
        # Make predictions for all test sequences
        all_predictions = []
        
        for x_seq in X_test:
            # Extract features
            features_dict = self._extract_global_features(x_seq)
            
            # Create test samples for all nodes and horizons
            X_test_samples = []
            
            for node_idx in range(self.n_nodes):
                for horizon in range(self.forecast_horizon):
                    features = features_dict.copy()
                    features['node_idx'] = node_idx
                    features['horizon'] = horizon
                    X_test_samples.append(features)
            
            # Convert to DataFrame
            X_test_df = pd.DataFrame(X_test_samples)
            
            # Make predictions
            dtest = xgb.DMatrix(X_test_df)
            y_pred = self.model.predict(dtest)
            
            # Reshape predictions to [forecast_horizon, n_nodes]
            y_pred = y_pred.reshape(self.n_nodes, self.forecast_horizon).T
            
            all_predictions.append(y_pred)
        
        return np.array(all_predictions)
    
    def _extract_global_features(self, x_seq):
        """
        Extract global features for XGBoost model
        
        Args:
            x_seq: Input sequence
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Get sequence length and number of nodes
        seq_len = len(x_seq['voltage'])
        n_nodes = x_seq['voltage'].shape[1]
        
        # 1. Grid-level statistics from voltage
        voltage = x_seq['voltage']  # Shape: [seq_len, n_nodes]
        
        # Global statistics across all nodes
        for i in range(seq_len):
            features[f'voltage_global_mean_t{i}'] = np.mean(voltage[i])
            features[f'voltage_global_std_t{i}'] = np.std(voltage[i])
            features[f'voltage_global_min_t{i}'] = np.min(voltage[i])
            features[f'voltage_global_max_t{i}'] = np.max(voltage[i])
        
        # 2. Recent voltage patterns (last few timesteps for each node)
        for t in range(max(0, seq_len-6), seq_len):
            for n in range(min(10, n_nodes)):  # Limit to first 10 nodes to avoid feature explosion
                features[f'voltage_node{n}_t{t}'] = voltage[t, n]
        
        # 3. Summary statistics across time for each node
        for n in range(min(10, n_nodes)):
            node_voltage = voltage[:, n]
            features[f'voltage_node{n}_mean'] = np.mean(node_voltage)
            features[f'voltage_node{n}_std'] = np.std(node_voltage)
            features[f'voltage_node{n}_trend'] = node_voltage[-1] - node_voltage[0]  # Overall trend
        
        # 4. Load and generation features
        load = x_seq['load']  # Shape: [seq_len, n_nodes, 2]
        gen = x_seq['gen']    # Shape: [seq_len, n_nodes, 2]
        
        # Last timestep load and generation for first few nodes
        for n in range(min(10, n_nodes)):
            features[f'load_p_node{n}'] = load[-1, n, 0]
            features[f'load_q_node{n}'] = load[-1, n, 1]
            features[f'gen_p_node{n}'] = gen[-1, n, 0]
            features[f'gen_q_node{n}'] = gen[-1, n, 1]
        
        # Global load and generation
        features['total_load_p'] = np.sum(load[-1, :, 0])
        features['total_load_q'] = np.sum(load[-1, :, 1])
        features['total_gen_p'] = np.sum(gen[-1, :, 0])
        features['total_gen_q'] = np.sum(gen[-1, :, 1])
        features['load_gen_p_ratio'] = features['total_load_p'] / (features['total_gen_p'] + 1e-10)
        
        # 5. Time features (from last timestep)
        for i, feat in enumerate(x_seq['time_features'][-1]):
            features[f'time_feature_{i}'] = feat
        
        return features

# Main forecasting experiment function
def global_xgboost_forecasting(builder, max_timestamps=480, look_back=48, forecast_horizon=48, stride=24):
    """
    Power grid forecasting with global XGBoost model
    
    Args:
        builder: PandaPowerFlowBuilder instance
        max_timestamps: Maximum number of timestamps to use
        look_back: Number of past timesteps to use
        forecast_horizon: Number of future timesteps to predict
        stride: Step size between sequences
    
    Returns:
        Dictionary with results and models
    """
    print(f"Running global XGBoost forecasting with look_back={look_back}, forecast_horizon={forecast_horizon}, stride={stride}")
    
    # Extract timestamps
    timestamps = builder.timestamps
    if max_timestamps and len(timestamps) > max_timestamps:
        timestamps = timestamps[:max_timestamps]
        print(f"Using a subset of {len(timestamps)} timestamps out of {len(builder.timestamps)}")
    else:
        print(f"Using all {len(timestamps)} timestamps")
    
    # Process the data
    voltage_data = []  # Voltage values for each timestep
    load_data = []     # Load values for each timestep
    gen_data = []      # Generation values for each timestep
    time_features = [] # Temporal features
    node_ids = []      # Bus IDs
    
    print("Extracting power grid data...")
    for t, ts in enumerate(timestamps):
        if t % 50 == 0:
            print(f"  Processing timestamp {t}/{len(timestamps)}")
        
        # Extract datetime features
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        hour_sin = np.sin(2 * np.pi * dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * dt.hour / 24)
        day_sin = np.sin(2 * np.pi * dt.weekday() / 7)
        day_cos = np.cos(2 * np.pi * dt.weekday() / 7)
        month_sin = np.sin(2 * np.pi * dt.month / 12)
        month_cos = np.cos(2 * np.pi * dt.month / 12)
        time_features.append([hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos])
        
        # Run the model for this timestamp
        sample = builder.run(ts)
        
        # Extract bus IDs on first iteration
        if t == 0:
            node_ids = sorted(sample.bus.index)
            
        # Extract values for each bus
        vm_values = []  # Voltage magnitude
        load_p_values = []  # Active power load
        load_q_values = []  # Reactive power load
        gen_p_values = []  # Active power generation
        gen_q_values = []  # Reactive power generation
        
        for idx in node_ids:
            # Voltage values
            if idx in sample.res_bus.index:
                vm_values.append(float(sample.res_bus.loc[idx]['vm_pu']))
                
                # Initialize load and generation to zero
                load_p = 0.0
                load_q = 0.0
                gen_p = 0.0
                gen_q = 0.0
                
                # Add load contributions
                for load_idx, load in sample.load.iterrows():
                    if load['bus'] == idx:
                        load_p += float(load['p_mw'])
                        load_q += float(load['q_mvar'])
                
                # Add generation contributions
                for gen_idx, gen in sample.gen.iterrows():
                    if gen['bus'] == idx:
                        if gen_idx in sample.res_gen.index:
                            gen_p += float(sample.res_gen.loc[gen_idx]['p_mw'])
                            gen_q += float(sample.res_gen.loc[gen_idx]['q_mvar'])
                
                load_p_values.append(load_p)
                load_q_values.append(load_q)
                gen_p_values.append(gen_p)
                gen_q_values.append(gen_q)
            else:
                # Default values if bus not found
                vm_values.append(1.0)
                load_p_values.append(0.0)
                load_q_values.append(0.0)
                gen_p_values.append(0.0)
                gen_q_values.append(0.0)
                
        voltage_data.append(np.array(vm_values))
        load_data.append(np.array([load_p_values, load_q_values]).T)
        gen_data.append(np.array([gen_p_values, gen_q_values]).T)
    
    voltage_data = np.array(voltage_data)
    load_data = np.array(load_data)
    gen_data = np.array(gen_data)
    time_features = np.array(time_features)
    
    print(f"Data shapes:")
    print(f"  Voltage data: {voltage_data.shape}")
    print(f"  Load data: {load_data.shape}")
    print(f"  Generation data: {gen_data.shape}")
    print(f"  Time features: {time_features.shape}")
    
    # Normalize the data
    print("Normalizing data...")
    voltage_scaler = StandardScaler()
    load_scaler = StandardScaler()
    gen_scaler = StandardScaler()
    
    # Reshape for scalers
    voltage_data_norm = voltage_scaler.fit_transform(voltage_data.reshape(-1, 1)).reshape(voltage_data.shape)
    load_data_flat = load_data.reshape(-1, load_data.shape[-1])
    load_data_norm = load_scaler.fit_transform(load_data_flat).reshape(load_data.shape)
    gen_data_flat = gen_data.reshape(-1, gen_data.shape[-1])
    gen_data_norm = gen_scaler.fit_transform(gen_data_flat).reshape(gen_data.shape)
    
    # Create sequences for multi-step forecasting
    X_data = []
    y_data = []
    
    print("Creating forecasting sequences...")
    for t in range(0, len(timestamps) - look_back - forecast_horizon + 1, stride):
        # Input sequence: past values
        x_seq = {
            'voltage': voltage_data_norm[t:t+look_back],
            'load': load_data_norm[t:t+look_back],
            'gen': gen_data_norm[t:t+look_back],
            'time_features': time_features[t:t+look_back],
            'voltage_raw': voltage_data[t:t+look_back]  # Keep raw values for naive models
        }
        
        # Target: future voltage values (normalized)
        y_seq = voltage_data_norm[t+look_back:t+look_back+forecast_horizon]
        
        X_data.append(x_seq)
        y_data.append(y_seq)
    
    print(f"Created {len(X_data)} sequences for training and evaluation")
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(X_data))
    val_size = int(0.15 * len(X_data))
    
    X_train = X_data[:train_size]
    y_train = y_data[:train_size]
    X_val = X_data[train_size:train_size+val_size]
    y_val = y_data[train_size:train_size+val_size]
    X_test = X_data[train_size+val_size:]
    y_test = y_data[train_size+val_size:]
    
    print(f"Train set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # Train and evaluate models
    models = {}
    
    # 1. Naive model (persistence forecast)
    print("\nTraining Naive model...")
    models['Naive'] = NaiveModel(forecast_horizon)
    
    # 2. Mean model (average of historical values)
    print("\nTraining Mean model...")
    models['Mean'] = MeanModel(forecast_horizon)
    
    # 3. Global XGBoost model
    if HAS_XGBOOST:
        print("\nTraining Global XGBoost model...")
        n_nodes = voltage_data.shape[1]
        xgb_model = GlobalXGBoostModel(n_nodes, forecast_horizon)
        xgb_model.fit(X_train, y_train, X_val, y_val)
        models['XGBoost'] = xgb_model
    
    # Convert test data to original scale for evaluation
    y_test_raw = []
    for i, x in enumerate(X_test):
        # Get the indices for this test sample
        start_idx = train_size + val_size + i + look_back
        end_idx = start_idx + forecast_horizon
        
        # Extract raw voltage values for these indices
        if end_idx <= len(voltage_data):
            y_horizon = voltage_data[start_idx:end_idx]
            y_test_raw.append(y_horizon)
    
    y_test_raw = np.array(y_test_raw)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name} model...")
        
        # Make predictions
        y_pred = []
        for x in X_test:
            pred = model.predict(x)
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        
        # For ML models, convert back to original scale
        if name == 'XGBoost':
            # Inverse transform predictions
            y_pred_shape = y_pred.shape
            y_pred_flat = y_pred.reshape(-1, 1)
            y_pred_flat = voltage_scaler.inverse_transform(y_pred_flat)
            y_pred = y_pred_flat.reshape(y_pred_shape)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_raw.reshape(-1), y_pred.reshape(-1))
        rmse = np.sqrt(mean_squared_error(y_test_raw.reshape(-1), y_pred.reshape(-1)))
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred
        }
        
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
    
    # Plot comparison of overall performance
    plt.figure(figsize=(10, 6))
    
    model_names = list(results.keys())
    mae_values = [results[name]['mae'] for name in model_names]
    
    plt.bar(model_names, mae_values)
    plt.ylabel('Mean Absolute Error')
    plt.title('Overall Model Performance Comparison')
    plt.grid(axis='y')
    
    # Plot comparison by forecast horizon
    plt.figure(figsize=(12, 6))
    
    for name, result in results.items():
        # Calculate MAE for each horizon
        mae_by_horizon = []
        for h in range(forecast_horizon):
            horizon_mae = mean_absolute_error(
                y_test_raw[:, h, :].flatten(), 
                result['predictions'][:, h, :].flatten()
            )
            mae_by_horizon.append(horizon_mae)
        
        plt.plot(range(1, forecast_horizon + 1), mae_by_horizon, marker='o', label=name)
    
    plt.xlabel('Forecast Horizon (timesteps)')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Performance by Forecast Horizon')
    plt.legend()
    plt.grid(True)
    
    # Plot a sample node's voltage prediction
    node_idx = 0  # Change this to visualize different nodes
    sample_idx = 0  # Change this to visualize different test samples
    
    plt.figure(figsize=(12, 6))
    # Plot actual values
    plt.plot(range(len(y_test_raw[sample_idx])), 
             y_test_raw[sample_idx][:, node_idx], 
             'b-', marker='o', label='Actual')
    
    # Plot predictions for each model
    for name, result in results.items():
        plt.plot(range(len(result['predictions'][sample_idx])), 
                 result['predictions'][sample_idx][:, node_idx], 
                 '--', marker='x', label=f'{name} Prediction')
    
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Voltage (p.u.)')
    plt.title(f'Voltage Prediction for Node {node_ids[node_idx]}')
    plt.legend()
    plt.grid(True)
    
    return {
        'models': models,
        'results': results,
        'node_ids': node_ids,
        'data': {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test_raw
        },
        'scalers': {
            'voltage': voltage_scaler,
            'load': load_scaler,
            'gen': gen_scaler
        }
    }

# Run the experiment if executed directly
if __name__ == "__main__":
    try:
        save_dir = "../data/parsed/"
        load_path = os.path.join(save_dir, "builder.pkl")
        
        # Add parent directory to path to handle module dependencies
        sys.path.append('../')
        with open(load_path, "rb") as f:
            builder = pickle.load(f)
        
        # Run the experiment
        print("Starting global XGBoost power grid forecasting...")
        results = global_xgboost_forecasting(
            builder,
            max_timestamps=480,
            look_back=48,    # Extended look-back window
            forecast_horizon=48,  # Extended forecast horizon
            stride=24
        )
        
        # Save results
        save_results_path = os.path.join(save_dir, "global_xgboost_results.pkl")
        with open(save_results_path, "wb") as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to {save_results_path}")
        
        plt.show()
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print("Please check your file paths and module dependencies.")