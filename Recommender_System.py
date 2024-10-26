"""
Hybrid Recommendation System
Combines collaborative filtering, content-based filtering, and deep learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from tqdm import tqdm

# Install: pip install torch pandas scikit-learn scipy


class MatrixFactorization(nn.Module):
    """Matrix Factorization for collaborative filtering"""
    
    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Dot product
        dot_product = (user_emb * item_emb).sum(dim=1, keepdim=True)
        
        # Add biases
        prediction = (dot_product + 
                     self.user_bias(user_ids) + 
                     self.item_bias(item_ids) + 
                     self.global_bias)
        
        return prediction.squeeze()


class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering (NCF)"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super().__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        return output.squeeze()


class RatingDataset(Dataset):
    """Dataset for user-item ratings"""
    
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class ContentBasedRecommender:
    """Content-based filtering using item features"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.item_profiles = None
        self.item_similarity = None
    
    def fit(self, item_features):
        """
        Fit on item features (descriptions, genres, etc.)
        item_features: list of text descriptions for each item
        """
        self.item_profiles = self.vectorizer.fit_transform(item_features)
        self.item_similarity = cosine_similarity(self.item_profiles)
    
    def recommend(self, item_id, top_k=10):
        """Recommend similar items based on content"""
        if self.item_similarity is None:
            raise ValueError("Model not fitted yet!")
        
        similarities = self.item_similarity[item_id]
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        return similar_indices, similarities[similar_indices]


class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches"""
    
    def __init__(self, num_users, num_items, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        
        # Collaborative filtering models
        self.mf_model = MatrixFactorization(num_users, num_items).to(device)
        self.ncf_model = NeuralCollaborativeFiltering(num_users, num_items).to(device)
        
        # Content-based model
        self.content_model = ContentBasedRecommender()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_rmse = []
    
    def train_collaborative(self, train_loader, val_loader, model_type='ncf', 
                          epochs=20, lr=0.001):
        """Train collaborative filtering model"""
        
        if model_type == 'mf':
            model = self.mf_model
        else:
            model = self.ncf_model
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        
        best_rmse = float('inf')
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            
            for user_ids, item_ids, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0
            all_preds = []
            all_ratings = []
            
            with torch.no_grad():
                for user_ids, item_ids, ratings in val_loader:
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)
                    
                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    
                    val_loss += loss.item()
                    all_preds.extend(predictions.cpu().numpy())
                    all_ratings.extend(ratings.cpu().numpy())
            
            val_loss /= len(val_loader)
            rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_ratings)) ** 2))
            
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_rmse.append(rmse)
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, RMSE = {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(model.state_dict(), f'best_{model_type}_model.pth')
                print(f"Saved best model with RMSE: {best_rmse:.4f}")
    
    def fit_content_based(self, item_features):
        """Fit content-based model"""
        self.content_model.fit(item_features)
    
    def predict_rating(self, user_id, item_id, model_type='ncf'):
        """Predict rating for a user-item pair"""
        if model_type == 'mf':
            model = self.mf_model
        else:
            model = self.ncf_model
        
        model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id]).to(self.device)
            item_tensor = torch.LongTensor([item_id]).to(self.device)
            prediction = model(user_tensor, item_tensor)
        
        return prediction.item()
    
    def recommend_collaborative(self, user_id, top_k=10, model_type='ncf', 
                              exclude_items=None):
        """Recommend items using collaborative filtering"""
        if model_type == 'mf':
            model = self.mf_model
        else:
            model = self.ncf_model
        
        model.eval()
        
        # Predict ratings for all items
        all_predictions = []
        
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * self.num_items).to(self.device)
            item_tensor = torch.LongTensor(range(self.num_items)).to(self.device)
            predictions = model(user_tensor, item_tensor)
            all_predictions = predictions.cpu().numpy()
        
        # Exclude already rated items
        if exclude_items is not None:
            all_predictions[exclude_items] = -np.inf
        
        # Get top-k recommendations
        top_indices = np.argsort(all_predictions)[::-1][:top_k]
        top_scores = all_predictions[top_indices]
        
        return top_indices, top_scores
    
    def recommend_hybrid(self, user_id, top_k=10, cf_weight=0.7, cb_weight=0.3,
                        user_history=None, item_features_available=True):
        """
        Hybrid recommendations combining collaborative and content-based
        cf_weight: weight for collaborative filtering
        cb_weight: weight for content-based filtering
        """
        # Get collaborative filtering recommendations
        cf_items, cf_scores = self.recommend_collaborative(
            user_id, top_k=top_k*2, exclude_items=user_history
        )
        
        # Combine with content-based if available
        if item_features_available and user_history is not None and len(user_history) > 0:
            # Get content-based recommendations based on user history
            cb_scores = np.zeros(self.num_items)
            
            for hist_item in user_history[-5:]:  # Use last 5 items
                similar_items, similarities = self.content_model.recommend(
                    hist_item, top_k=self.num_items
                )
                cb_scores[similar_items] += similarities
            
            # Normalize content-based scores
            if cb_scores.max() > 0:
                cb_scores = cb_scores / cb_scores.max()
            
            # Combine scores
            hybrid_scores = np.zeros(self.num_items)
            hybrid_scores[cf_items] = cf_weight * (cf_scores / cf_scores.max())
            hybrid_scores += cb_weight * cb_scores
            
            # Exclude user history
            if user_history is not None:
                hybrid_scores[user_history] = -np.inf
            
            # Get top-k
            top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
            top_scores = hybrid_scores[top_indices]
        else:
            # Fall back to collaborative filtering only
            top_indices = cf_items[:top_k]
            top_scores = cf_scores[:top_k]
        
        return top_indices, top_scores
    
    def evaluate(self, test_loader, model_type='ncf'):
        """Evaluate model on test set"""
        if model_type == 'mf':
            model = self.mf_model
        else:
            model = self.ncf_model
        
        model.eval()
        all_preds = []
        all_ratings = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                
                predictions = model(user_ids, item_ids)
                all_preds.extend(predictions.cpu().numpy())
                all_ratings.extend(ratings.numpy())
        
        all_preds = np.array(all_preds)
        all_ratings = np.array(all_ratings)
        
        rmse = np.sqrt(np.mean((all_preds - all_ratings) ** 2))
        mae = np.mean(np.abs(all_preds - all_ratings))
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        
        return rmse, mae
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.val_rmse, label='Val RMSE', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Validation RMSE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('recommender_training.png', dpi=300)
        plt.show()


# Generate synthetic data
def generate_synthetic_ratings(num_users=1000, num_items=500, num_ratings=50000):
    """Generate synthetic user-item ratings"""
    np.random.seed(42)
    
    user_ids = np.random.randint(0, num_users, num_ratings)
    item_ids = np.random.randint(0, num_items, num_ratings)
    
    # Generate ratings with some structure
    user_preferences = np.random.rand(num_users, 10)
    item_features = np.random.rand(num_items, 10)
    
    ratings = []
    for user_id, item_id in zip(user_ids, item_ids):
        # Base rating from user-item interaction
        base_rating = np.dot(user_preferences[user_id], item_features[item_id]) * 5
        # Add noise
        noise = np.random.normal(0, 0.5)
        rating = np.clip(base_rating + noise, 1, 5)
        ratings.append(rating)
    
    return user_ids, item_ids, ratings


def generate_item_descriptions(num_items=500):
    """Generate synthetic item descriptions"""
    genres = ['action', 'comedy', 'drama', 'thriller', 'romance', 'sci-fi', 'horror']
    adjectives = ['amazing', 'exciting', 'boring', 'thrilling', 'emotional', 'funny', 'scary']
    
    descriptions = []
    for i in range(num_items):
        genre = np.random.choice(genres, size=2, replace=False)
        adj = np.random.choice(adjectives, size=3, replace=False)
        desc = f"A {adj[0]} {genre[0]} movie with {adj[1]} scenes and {adj[2]} {genre[1]} elements"
        descriptions.append(desc)
    
    return descriptions


if __name__ == "__main__":
    print("Generating synthetic rating data...")
    num_users = 1000
    num_items = 500
    
    user_ids, item_ids, ratings = generate_synthetic_ratings(num_users, num_items, num_ratings=50000)
    
    # Split data
    train_size = int(0.8 * len(ratings))
    val_size = int(0.1 * len(ratings))
    
    train_users = user_ids[:train_size]
    train_items = item_ids[:train_size]
    train_ratings = ratings[:train_size]
    
    val_users = user_ids[train_size:train_size+val_size]
    val_items = item_ids[train_size:train_size+val_size]
    val_ratings = ratings[train_size:train_size+val_size]
    
    test_users = user_ids[train_size+val_size:]
    test_items = item_ids[train_size+val_size:]
    test_ratings = ratings[train_size+val_size:]
    
    # Create datasets
    train_dataset = RatingDataset(train_users, train_items, train_ratings)
    val_dataset = RatingDataset(val_users, val_items, val_ratings)
    test_dataset = RatingDataset(test_users, test_items, test_ratings)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize hybrid recommender
    print("\nInitializing Hybrid Recommender System...")
    recommender = HybridRecommender(num_users, num_items)
    
    # Train collaborative filtering
    print("\nTraining Neural Collaborative Filtering model...")
    recommender.train_collaborative(train_loader, val_loader, model_type='ncf', epochs=20, lr=0.001)
    
    # Plot training history
    recommender.plot_training_history()
    
    # Evaluate
    print("\nEvaluating on test set...")
    recommender.evaluate(test_loader, model_type='ncf')
    
    # Train content-based model
    print("\nTraining content-based model...")
    item_descriptions = generate_item_descriptions(num_items)
    recommender.fit_content_based(item_descriptions)
    
    # Generate recommendations
    print("\n" + "="*50)
    print("SAMPLE RECOMMENDATIONS")
    print("="*50)
    
    test_user_id = 42
    user_history = [10, 25, 30, 45]  # Simulated user history
    
    # Collaborative filtering recommendations
    print(f"\nCollaborative Filtering Recommendations for User {test_user_id}:")
    cf_items, cf_scores = recommender.recommend_collaborative(
        test_user_id, top_k=10, exclude_items=user_history
    )
    for i, (item, score) in enumerate(zip(cf_items, cf_scores), 1):
        print(f"{i}. Item {item} (Score: {score:.3f})")
    
    # Hybrid recommendations
    print(f"\nHybrid Recommendations for User {test_user_id}:")
    hybrid_items, hybrid_scores = recommender.recommend_hybrid(
        test_user_id, top_k=10, user_history=user_history
    )
    for i, (item, score) in enumerate(zip(hybrid_items, hybrid_scores), 1):
        print(f"{i}. Item {item} (Score: {score:.3f})")
    
    print("\nRecommender system training complete!")
    print("Models saved: 'best_ncf_model.pth', 'best_mf_model.pth'")