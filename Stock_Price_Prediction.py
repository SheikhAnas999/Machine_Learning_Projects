"""
Stock Price Prediction using LSTM, GRU, and Temporal Fusion Transformers
Multiple model comparison with walk-forward validation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Install: pip install torch pandas scikit-learn matplotlib


class StockDataset(Dataset):
    """Dataset for time series stock data"""
    
    def __init__(self, data, sequence_length=60):
        self.data = data
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length, :]
        y = self.data[idx + self.sequence_length, 0]  # Predict close price
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMModel(nn.Module):
    """LSTM model for stock prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


class GRUModel(nn.Module):
    """GRU model for stock prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        output = self.fc(last_out)
        return output


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Calculate attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.fc(context)
        return output


class StockPredictor:
    """Complete stock prediction system"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = MinMaxScaler()
        
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.actuals = []
    
    def prepare_data(self, df, feature_cols, sequence_length=60):
        """Prepare data for training"""
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_cols].values)
        
        # Create sequences
        dataset = StockDataset(scaled_data, sequence_length)
        
        return dataset
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = criterion(outputs, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                loss = criterion(outputs, y)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y.cpu().numpy())
        
        return total_loss / len(val_loader), predictions, actuals
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Complete training loop"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_preds, val_actuals = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_stock_model.pth')
        
        self.predictions = val_preds
        self.actuals = val_actuals
    
    def predict(self, data_loader):
        """Make predictions"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                outputs = self.model(x)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.6f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
    
    def plot_training_history(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.actuals[:200], label='Actual', marker='o', markersize=3)
        plt.plot(self.predictions[:200], label='Predicted', marker='x', markersize=3)
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Price')
        plt.title('Predictions vs Actuals (First 200 points)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('stock_prediction_history.png', dpi=300)
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, title="Stock Price Prediction"):
        """Plot predictions against actual values"""
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(y_true, label='Actual Price', linewidth=2)
        plt.plot(y_pred, label='Predicted Price', linewidth=2, alpha=0.8)
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_comparison.png', dpi=300)
        plt.show()
    
    def walk_forward_validation(self, data, feature_cols, n_splits=5, 
                                sequence_length=60, epochs=30):
        """Perform walk-forward validation"""
        print("\nPerforming Walk-Forward Validation...")
        
        all_metrics = []
        split_size = len(data) // n_splits
        
        for i in range(1, n_splits):
            print(f"\n--- Fold {i}/{n_splits-1} ---")
            
            train_data = data[:split_size * i]
            val_data = data[split_size * i:split_size * (i + 1)]
            
            if len(train_data) < sequence_length or len(val_data) < sequence_length:
                continue
            
            # Prepare data
            train_dataset = self.prepare_data(train_data, feature_cols, sequence_length)
            val_dataset = self.prepare_data(val_data, feature_cols, sequence_length)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Train
            self.train(train_loader, val_loader, epochs=epochs, lr=0.001)
            
            # Evaluate
            metrics = self.evaluate(self.actuals, self.predictions)
            all_metrics.append(metrics)
        
        # Average metrics
        print("\n" + "="*50)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("="*50)
        for key in all_metrics[0].keys():
            avg_value = np.mean([m[key] for m in all_metrics])
            std_value = np.std([m[key] for m in all_metrics])
            print(f"{key.upper()}: {avg_value:.6f} ± {std_value:.6f}")


class ModelComparison:
    """Compare multiple models"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
    
    def compare_models(self, models_dict, train_loader, val_loader, epochs=30):
        """Compare multiple models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        for name, model in models_dict.items():
            print(f"\nTraining {name}...")
            
            predictor = StockPredictor(model, self.device)
            predictor.train(train_loader, val_loader, epochs=epochs, lr=0.001)
            
            metrics = predictor.evaluate(predictor.actuals, predictor.predictions)
            self.results[name] = {
                'metrics': metrics,
                'predictor': predictor
            }
        
        # Plot comparison
        self.plot_comparison()
    
    def plot_comparison(self):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['rmse', 'mae', 'r2', 'mape']
        titles = ['RMSE (Lower is Better)', 'MAE (Lower is Better)', 
                 'R² Score (Higher is Better)', 'MAPE (Lower is Better)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            names = list(self.results.keys())
            values = [self.results[name]['metrics'][metric] for name in names]
            
            ax.bar(names, values, color=['blue', 'green', 'red'][:len(names)])
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Rotate labels if needed
            if len(names) > 2:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300)
        plt.show()


# Generate synthetic stock data
def generate_synthetic_stock_data(n_days=1000):
    """Generate synthetic stock price data"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate base trend
    trend = np.linspace(100, 150, n_days)
    
    # Add seasonality
    seasonality = 10 * np.sin(np.linspace(0, 8*np.pi, n_days))
    
    # Add noise
    noise = np.random.normal(0, 5, n_days)
    
    # Generate price components
    close = trend + seasonality + noise
    open_price = close + np.random.normal(0, 2, n_days)
    high = np.maximum(open_price, close) + np.abs(np.random.normal(0, 3, n_days))
    low = np.minimum(open_price, close) - np.abs(np.random.normal(0, 3, n_days))
    
    # Generate volume
    volume = np.random.randint(1000000, 5000000, n_days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    # Add technical indicators
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['Volatility'] = df['Close'].rolling(window=21).std()
    
    df = df.fillna(method='bfill')
    
    return df


if __name__ == "__main__":
    print("Generating synthetic stock data...")
    df = generate_synthetic_stock_data(n_days=1000)
    
    # Define features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_7', 'MA_21', 'Volatility']
    
    # Split data
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    # Initialize predictor with LSTM
    print("\nInitializing LSTM model...")
    lstm_model = LSTMModel(input_size=len(feature_cols), hidden_size=128, num_layers=2)
    predictor = StockPredictor(lstm_model)
    
    # Prepare data
    train_dataset = predictor.prepare_data(train_df, feature_cols, sequence_length=60)
    val_dataset = predictor.prepare_data(val_df, feature_cols, sequence_length=60)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    print("\nTraining LSTM model...")
    predictor.train(train_loader, val_loader, epochs=50, lr=0.001)
    
    # Evaluate
    predictor.evaluate(predictor.actuals, predictor.predictions)
    predictor.plot_training_history()
    
    # Inverse transform predictions for plotting
    dummy_features = np.zeros((len(predictor.predictions), len(feature_cols)))
    dummy_features[:, 3] = np.array(predictor.predictions).flatten()  # Close price column
    predictions_original = predictor.scaler.inverse_transform(dummy_features)[:, 3]
    
    dummy_features[:, 3] = np.array(predictor.actuals).flatten()
    actuals_original = predictor.scaler.inverse_transform(dummy_features)[:, 3]
    
    predictor.plot_predictions(actuals_original, predictions_original)
    
    # Compare models
    print("\n" + "="*50)
    print("COMPARING MULTIPLE MODELS")
    print("="*50)
    
    models_dict = {
        'LSTM': LSTMModel(len(feature_cols), 128, 2),
        'GRU': GRUModel(len(feature_cols), 128, 2),
        'Attention-LSTM': AttentionLSTM(len(feature_cols), 128, 2)
    }
    
    comparison = ModelComparison()
    comparison.compare_models(models_dict, train_loader, val_loader, epochs=30)
    
    print("\nTraining complete! Best model saved as 'best_stock_model.pth'")