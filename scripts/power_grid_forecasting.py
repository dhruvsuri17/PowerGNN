import networkx as nx
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class TemporalGNN(torch.nn.Module):
    """
    Temporal Graph Neural Network for voltage prediction in power networks.
    Combines GNN layers with temporal processing to capture both spatial and temporal dependencies.
    """
    def __init__(self, node_features, edge_features, hidden_channels=64, num_layers=3, forecast_horizon=24):
        super(TemporalGNN, self).__init__()
        
        self.forecast_horizon = forecast_horizon
        
        # GNN layers for spatial dependencies
        self.convs = torch.nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(node_features, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(pyg_nn.GCNConv(hidden_channels, hidden_channels))
            
        # LSTM for temporal processing
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        
        # Final prediction layers - modified to output forecast_horizon values
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, forecast_horizon)  # Predict voltage for multiple horizons
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, data_seq, edge_index, edge_attr):
        """
        Forward pass through the temporal GNN.
        
        Args:
            data_seq: Tensor of shape [batch_size, seq_len, num_nodes, node_features]
            edge_index: Edge indices
            edge_attr: Edge features
            
        Returns:
            Predicted voltage magnitudes for each node across forecast_horizon
        """
        batch_size, seq_len, num_nodes, node_features = data_seq.shape
        
        # Process each graph in the sequence with GNN
        seq_output = []
        for t in range(seq_len):
            x = data_seq[:, t, :, :]  # [batch_size, num_nodes, node_features]
            x = x.reshape(-1, node_features)  # [batch_size*num_nodes, node_features]
            
            # Apply GNN layers
            for conv in self.convs:
                x = self.relu(conv(x, edge_index, edge_attr))
            
            # Reshape back
            x = x.reshape(batch_size, num_nodes, -1)  # [batch_size, num_nodes, hidden_channels]
            seq_output.append(x)
            
        # Stack outputs for LSTM processing
        lstm_input = torch.stack(seq_output, dim=1)  # [batch_size, seq_len, num_nodes, hidden_channels]
        
        # Process each node's sequence through LSTM
        lstm_out = []
        for node_idx in range(num_nodes):
            node_seq = lstm_input[:, :, node_idx, :]  # [batch_size, seq_len, hidden_channels]
            out, _ = self.lstm(node_seq)
            lstm_out.append(out[:, -1, :])  # Take the last output
            
        lstm_out = torch.stack(lstm_out, dim=1)  # [batch_size, num_nodes, hidden_channels]
        
        # Final prediction layers
        x = self.relu(self.lin1(lstm_out))
        x = self.lin2(x)  # [batch_size, num_nodes, forecast_horizon]
        
        return x

class NaiveModel:
    """
    Naive forecasting model that uses the last observed value as the prediction
    for all future time steps.
    """
    def __init__(self, forecast_horizon=24):
        self.forecast_horizon = forecast_horizon
    
    def fit(self, X_data, y_data):
        # No training needed for naive model
        pass
    
    def predict(self, X_data):
        predictions = []
        for seq in X_data:
            # Use the last observed value for each node as prediction
            last_values = seq[-1]['node_vm_pu']  # Extract voltage values from last timestep
            # Repeat the last value for each forecast horizon
            forecast = np.tile(last_values, (self.forecast_horizon, 1)).T
            predictions.append(forecast)
        
        return np.array(predictions)

class StatisticalModel:
    """
    Statistical forecasting model using ARIMA or Exponential Smoothing
    for each node in the power network.
    """
    def __init__(self, forecast_horizon=24, method='exp_smoothing'):
        self.forecast_horizon = forecast_horizon
        self.method = method
        self.models = {}  # Store a model for each node
    
    def fit(self, X_data, y_data):
        # Extract time series for each node
        n_nodes = len(X_data[0][0]['node_vm_pu'])
        
        for node_idx in range(n_nodes):
            # Extract historical voltage values for this node
            node_values = []
            for seq in X_data:
                for time_step in seq:
                    node_values.append(time_step['node_vm_pu'][node_idx])
            
            # Fit model based on selected method
            if self.method == 'arima':
                # Simple ARIMA model (p=1, d=0, q=0) for demonstration
                try:
                    model = ARIMA(node_values, order=(1, 0, 0))
                    fitted_model = model.fit()
                    self.models[node_idx] = fitted_model
                except:
                    # Fallback to simpler model if ARIMA fails
                    self.models[node_idx] = None
                    print(f"ARIMA failed for node {node_idx}, using naive forecasting instead")
            
            elif self.method == 'exp_smoothing':
                try:
                    # Simple Exponential Smoothing
                    model = ExponentialSmoothing(node_values)
                    fitted_model = model.fit()
                    self.models[node_idx] = fitted_model
                except:
                    # Fallback to simpler model if Exp Smoothing fails
                    self.models[node_idx] = None
                    print(f"Exp Smoothing failed for node {node_idx}, using naive forecasting instead")
    
    def predict(self, X_data):
        predictions = []
        n_nodes = len(X_data[0][0]['node_vm_pu'])
        
        for seq in X_data:
            forecast = np.zeros((n_nodes, self.forecast_horizon))
            
            for node_idx in range(n_nodes):
                # Extract historical values for this sequence
                node_values = [time_step['node_vm_pu'][node_idx] for time_step in seq]
                
                if node_idx in self.models and self.models[node_idx] is not None:
                    # Use fitted model to forecast
                    if self.method == 'arima':
                        model_forecast = self.models[node_idx].forecast(steps=self.forecast_horizon)
                        forecast[node_idx, :] = model_forecast
                    elif self.method == 'exp_smoothing':
                        model_forecast = self.models[node_idx].forecast(self.forecast_horizon)
                        forecast[node_idx, :] = model_forecast
                else:
                    # Fallback to naive forecasting
                    forecast[node_idx, :] = node_values[-1]
            
            predictions.append(forecast)
        
        return np.array(predictions)

class RandomForestModel:
    """
    Random Forest model for voltage prediction.
    Trains a separate RF model for each node and time horizon.
    """
    def __init__(self, forecast_horizon=24):
        self.forecast_horizon = forecast_horizon
        self.models = {}  # Dictionary to store models for each node and horizon
        self.feature_importances = {}
    
    def _extract_features(self, X_data):
        """Extract features from the input sequences"""
        features_list = []
        for seq in X_data:
            # Flatten the sequence data into features
            seq_features = []
            
            # Include all node features from the most recent time step
            latest_node_features = seq[-1]['node_features']
            seq_features.extend(latest_node_features.flatten())
            
            # Add temporal features from several recent time steps
            for t in range(max(0, len(seq)-3), len(seq)):
                vm_values = seq[t]['node_vm_pu']
                seq_features.extend(vm_values)
            
            features_list.append(seq_features)
        
        return np.array(features_list)
    
    def fit(self, X_data, y_data):
        # Extract features from input sequences
        X_features = self._extract_features(X_data)
        
        # Get number of nodes
        n_nodes = y_data[0].shape[0] if len(y_data) > 0 else 0
        
        # For each node and forecast horizon, train a separate RF model
        for node_idx in range(n_nodes):
            self.models[node_idx] = {}
            
            for horizon in range(self.forecast_horizon):
                # Extract target values for this node and horizon
                y_node_horizon = np.array([y[node_idx, horizon] for y in y_data])
                
                # Train Random Forest model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_features, y_node_horizon)
                
                # Save model and feature importances
                self.models[node_idx][horizon] = rf_model
                self.feature_importances[(node_idx, horizon)] = rf_model.feature_importances_
    
    def predict(self, X_data):
        # Extract features from input sequences
        X_features = self._extract_features(X_data)
        
        # Get number of nodes
        n_nodes = len(self.models.keys())
        
        # Initialize predictions array
        predictions = np.zeros((len(X_data), n_nodes, self.forecast_horizon))
        
        # Make predictions for each node and horizon
        for node_idx in range(n_nodes):
            for horizon in range(self.forecast_horizon):
                if node_idx in self.models and horizon in self.models[node_idx]:
                    model = self.models[node_idx][horizon]
                    predictions[:, node_idx, horizon] = model.predict(X_features)
                else:
                    # Use the last observed value if model is missing
                    for i, seq in enumerate(X_data):
                        predictions[i, node_idx, horizon] = seq[-1]['node_vm_pu'][node_idx]
        
        return predictions

def construct_enhanced_graph_sequence(builder, look_back=24, forecast_horizon=24, stride=24, normalize=True):
    """
    Constructs an enhanced sequence of graph data for voltage prediction with multi-step forecasting.
    
    Args:
        builder: PandaPowerFlowBuilder instance
        look_back: Number of past timesteps to include for temporal features
        forecast_horizon: Number of future timesteps to predict
        stride: Step size between sequences (default: equal to forecast_horizon for non-overlapping windows)
        normalize: Whether to normalize the features
        
    Returns:
        Tuple containing (X_data, y_data, feature_scalers, target_scaler)
        X_data: List of sequences of dicts containing node features, edge features, and graph structure
        y_data: Target voltage magnitudes for multiple forecast horizons
    """
    total_timesteps = len(builder.timestamps)
    
    # Extract temporal information from timestamps
    time_features = []
    for ts in builder.timestamps:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        # Create normalized time features
        hour_sin = np.sin(2 * np.pi * dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * dt.hour / 24)
        day_sin = np.sin(2 * np.pi * dt.weekday() / 7)
        day_cos = np.cos(2 * np.pi * dt.weekday() / 7)
        month_sin = np.sin(2 * np.pi * dt.month / 12)
        month_cos = np.cos(2 * np.pi * dt.month / 12)
        time_features.append([hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos])
    
    time_features = np.array(time_features)
    
    print(f"Building dataset with {total_timesteps} timestamps...")
    
    # Process each timestep and extract features
    node_features_list = []
    edge_features_list = []
    vm_pu_values = []  # Target voltage magnitudes
    graph_structures = []
    
    for t in range(total_timesteps):
        sample = builder.run(builder.timestamps[t])
        G = nx.Graph()
        
        # Add buses with enhanced features
        for idx, bus in sample.bus.iterrows():
            res_bus = sample.res_bus.loc[idx] if idx in sample.res_bus.index else None
            
            # Get connected loads
            load_p_mw = 0.0
            load_q_mvar = 0.0
            for load_idx, load in sample.load.iterrows():
                if load['bus'] == idx:
                    load_p_mw += float(load['p_mw'])
                    load_q_mvar += float(load['q_mvar'])
            
            # Add node with extensive features
            G.add_node(idx, 
                      vn_kv=float(bus['vn_kv']),
                      min_vm_pu=float(bus['min_vm_pu']),
                      max_vm_pu=float(bus['max_vm_pu']),
                      type=bus['type'],
                      zone=bus['zone'],
                      vm_pu=float(res_bus['vm_pu']) if res_bus is not None else 1.0,
                      va_degree=float(res_bus['va_degree']) if res_bus is not None else 0.0,
                      p_mw=float(res_bus['p_mw']) if res_bus is not None else 0.0,
                      q_mvar=float(res_bus['q_mvar']) if res_bus is not None else 0.0,
                      load_p_mw=load_p_mw,
                      load_q_mvar=load_q_mvar,
                      gen_p_mw=0.0,  # Will be updated if there's a generator
                      gen_q_mvar=0.0,  # Will be updated if there's a generator
                      hour_sin=time_features[t][0],
                      hour_cos=time_features[t][1],
                      day_sin=time_features[t][2],
                      day_cos=time_features[t][3],
                      month_sin=time_features[t][4],
                      month_cos=time_features[t][5])
        
        # Add edges (transmission lines)
        for idx, line in sample.line.iterrows():
            res_line = sample.res_line.loc[idx] if idx in sample.res_line.index else None
            G.add_edge(line['from_bus'], line['to_bus'],
                      r_ohm_per_km=float(line['r_ohm_per_km']),
                      x_ohm_per_km=float(line['x_ohm_per_km']),
                      length_km=float(line['length_km']),
                      c_nf_per_km=float(line['c_nf_per_km']),
                      max_i_ka=float(line['max_i_ka']),
                      p_from=float(res_line['p_from_mw']) if res_line is not None else 0.0,
                      q_from=float(res_line['q_from_mvar']) if res_line is not None else 0.0,
                      p_to=float(res_line['p_to_mw']) if res_line is not None else 0.0,
                      q_to=float(res_line['q_to_mvar']) if res_line is not None else 0.0,
                      loading=float(res_line['loading_percent']) if res_line is not None else 0.0)

        # Add transformers as edges
        for idx, trafo in sample.trafo.iterrows():
            res_trafo = sample.res_trafo.loc[idx] if idx in sample.res_trafo.index else None
            G.add_edge(trafo['hv_bus'], trafo['lv_bus'],
                      vk_percent=float(trafo['vk_percent']),
                      tap_pos=float(trafo['tap_pos']),
                      shift_degree=float(trafo['shift_degree']),
                      p_from=float(res_trafo['p_hv_mw']) if res_trafo is not None else 0.0,
                      q_from=float(res_trafo['q_hv_mvar']) if res_trafo is not None else 0.0,
                      p_to=float(res_trafo['p_lv_mw']) if res_trafo is not None else 0.0,
                      q_to=float(res_trafo['q_lv_mvar']) if res_trafo is not None else 0.0,
                      loading=float(res_trafo['loading_percent']) if res_trafo is not None else 0.0)

        # Add generators
        for idx, gen in sample.gen.iterrows():
            res_gen = sample.res_gen.loc[idx] if idx in sample.res_gen.index else None
            bus_idx = gen['bus']
            if bus_idx in G.nodes:
                G.nodes[bus_idx]['gen_p_mw'] = float(res_gen['p_mw']) if res_gen is not None else 0.0
                G.nodes[bus_idx]['gen_q_mvar'] = float(res_gen['q_mvar']) if res_gen is not None else 0.0
                G.nodes[bus_idx]['gen_min_p_mw'] = float(gen['min_p_mw'])
                G.nodes[bus_idx]['gen_max_p_mw'] = float(gen['max_p_mw'])
                G.nodes[bus_idx]['gen_min_q_mvar'] = float(gen['min_q_mvar'])
                G.nodes[bus_idx]['gen_max_q_mvar'] = float(gen['max_q_mvar'])
                G.nodes[bus_idx]['gen_vm_pu'] = float(gen['vm_pu'])
        
        # Extract node features (consistent order of nodes)
        sorted_nodes = sorted(G.nodes())
        
        # Extract voltage magnitude targets for prediction
        vm_pu = np.array([G.nodes[i]['vm_pu'] for i in sorted_nodes])
        vm_pu_values.append(vm_pu)
        
        # Node features excluding the target (voltage magnitude)
        node_features = []
        for i in sorted_nodes:
            features = [
                G.nodes[i]['vn_kv'], 
                G.nodes[i]['min_vm_pu'],
                G.nodes[i]['max_vm_pu'],
                G.nodes[i]['va_degree'],
                G.nodes[i]['p_mw'], 
                G.nodes[i]['q_mvar'],
                G.nodes[i]['load_p_mw'],
                G.nodes[i]['load_q_mvar'],
                G.nodes[i]['gen_p_mw'], 
                G.nodes[i]['gen_q_mvar'],
                G.nodes[i].get('gen_min_p_mw', 0.0),
                G.nodes[i].get('gen_max_p_mw', 0.0),
                G.nodes[i].get('gen_min_q_mvar', 0.0),
                G.nodes[i].get('gen_max_q_mvar', 0.0),
                G.nodes[i].get('gen_vm_pu', 1.0),
                G.nodes[i]['hour_sin'],
                G.nodes[i]['hour_cos'],
                G.nodes[i]['day_sin'],
                G.nodes[i]['day_cos'],
                G.nodes[i]['month_sin'],
                G.nodes[i]['month_cos']
            ]
            node_features.append(features)
        
        node_features_list.append(np.array(node_features))
        
        # Edge features
        edge_features = []
        edge_index = []
        
        # Process edges in a consistent order
        for u, v, d in sorted(G.edges(data=True), key=lambda x: (x[0], x[1])):
            u_idx = sorted_nodes.index(u)
            v_idx = sorted_nodes.index(v)
            
            # Create bidirectional edges for GNN
            edge_index.extend([[u_idx, v_idx], [v_idx, u_idx]])
            
            # Extract edge features
            if 'r_ohm_per_km' in d:  # Line
                e_feat = [
                    d['r_ohm_per_km'], 
                    d['x_ohm_per_km'], 
                    d['length_km'],
                    d.get('c_nf_per_km', 0.0),
                    d.get('max_i_ka', 0.0),
                    d['p_from'], 
                    d['q_from'], 
                    d['p_to'], 
                    d['q_to'], 
                    d['loading']
                ]
            else:  # Transformer
                e_feat = [
                    0.0,  # Placeholder for r_ohm_per_km
                    0.0,  # Placeholder for x_ohm_per_km
                    0.0,  # Placeholder for length_km
                    0.0,  # Placeholder for c_nf_per_km
                    0.0,  # Placeholder for max_i_ka
                    d['p_from'],
                    d['q_from'],
                    d['p_to'],
                    d['q_to'],
                    d['loading'],
                    d.get('vk_percent', 0.0),
                    d.get('tap_pos', 0.0),
                    d.get('shift_degree', 0.0)
                ]
            
            # Add edge features for both directions
            edge_features.extend([e_feat, e_feat])
        
        edge_features_list.append(np.array(edge_features))
        graph_structures.append((sorted_nodes, edge_index))
    
    # Normalize features if requested
    if normalize:
        node_scaler = StandardScaler()
        edge_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        # Flatten for scaling
        all_node_features = np.vstack(node_features_list)
        all_edge_features = np.vstack(edge_features_list)
        all_targets = np.concatenate(vm_pu_values)
        
        # Fit scalers
        node_scaler.fit(all_node_features)
        edge_scaler.fit(all_edge_features)
        target_scaler.fit(all_targets.reshape(-1, 1))
        
        # Scale features
        for i in range(len(node_features_list)):
            node_features_list[i] = node_scaler.transform(node_features_list[i])
            edge_features_list[i] = edge_scaler.transform(edge_features_list[i])
    else:
        node_scaler = None
        edge_scaler = None
        target_scaler = None
    
    # Create sequences for multi-step forecasting with specified stride
    X_data = []
    y_data = []
    
    for t in range(0, total_timesteps - look_back - forecast_horizon + 1, stride):
        # Input sequence
        input_seq = []
        for i in range(look_back):
            time_idx = t + i
            input_seq.append({
                'node_features': node_features_list[time_idx],
                'edge_features': edge_features_list[time_idx],
                'graph_structure': graph_structures[time_idx],
                'node_vm_pu': vm_pu_values[time_idx]  # Include actual voltage values for baseline models
            })
        
        # Target: voltage magnitudes for multiple future horizons
        target = np.zeros((len(vm_pu_values[0]), forecast_horizon))
        for h in range(forecast_horizon):
            target_idx = t + look_back + h
            if target_idx < total_timesteps:
                target[:, h] = vm_pu_values[target_idx]
        
        X_data.append(input_seq)
        y_data.append(target)
    
    scalers = {
        'node_scaler': node_scaler,
        'edge_scaler': edge_scaler,
        'target_scaler': target_scaler
    }
    
    return X_data, y_data, scalers, sorted_nodes

def prepare_batch_for_model(batch_data, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Prepares a batch of graph sequences for the GNN model.
    
    Args:
        batch_data: List of sequences of graph data
        device: PyTorch device
        
    Returns:
        Batched data ready for model input
    """
    batch_size = len(batch_data)
    seq_len = len(batch_data[0])
    
    # Assume all graphs have the same structure for simplicity
    num_nodes = batch_data[0][0]['node_features'].shape[0]
    node_features = batch_data[0][0]['node_features'].shape[1]
    
    # Prepare tensors
    x = torch.zeros((batch_size, seq_len, num_nodes, node_features), device=device)
    
    # Use the first graph's structure for all (assuming fixed topology)
    edge_index = torch.tensor(batch_data[0][0]['graph_structure'][1], dtype=torch.long, device=device).t().contiguous()
    edge_attr = torch.tensor(batch_data[0][0]['edge_features'], dtype=torch.float, device=device)
    
    # Fill node features for each sequence
    for b in range(batch_size):
        for s in range(seq_len):
            x[b, s] = torch.tensor(batch_data[b][s]['node_features'], dtype=torch.float, device=device)
    
    return x, edge_index, edge_attr

def train_voltage_predictor(X_train, y_train, X_val=None, y_val=None, forecast_horizon=24, epochs=50, batch_size=32):
    """
    Trains the Temporal GNN model for multi-step voltage prediction.
    
    Args:
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences
        y_val: Validation targets
        forecast_horizon: Number of timesteps to predict
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine dimensions
    node_features = X_train[0][0]['node_features'].shape[1]
    edge_features = X_train[0][0]['edge_features'].shape[1]
    
    # Initialize model
    model = TemporalGNN(node_features, edge_features, forecast_horizon=forecast_horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Create random batches
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = [X_train[idx] for idx in batch_indices]
            batch_y = torch.tensor([y_train[idx] for idx in batch_indices], dtype=torch.float, device=device)
            
            # Prepare batch
            x, edge_index, edge_attr = prepare_batch_for_model(batch_X, device)
            
            # Forward pass
            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr)
            
            # Calculate loss
            loss = criterion(out, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / (len(indices) // batch_size + 1)
        
        # Validation
        if X_val and y_val:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    end_idx = min(i + batch_size, len(X_val))
                    batch_X = X_val[i:end_idx]
                    batch_y = torch.tensor(y_val[i:end_idx], dtype=torch.float, device=device)
                    
                    # Prepare batch
                    x, edge_index, edge_attr = prepare_batch_for_model(batch_X, device)
                    
                    # Forward pass
                    out = model(x, edge_index, edge_attr)
                    loss = criterion(out, batch_y)
                    val_loss += loss.item() * (end_idx - i)
            
            avg_val_loss = val_loss / len(X_val)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")
    
    # Load best model if available
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model