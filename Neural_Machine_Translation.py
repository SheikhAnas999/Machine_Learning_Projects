"""
Neural Machine Translation using Transformer Architecture
Implements attention mechanism, beam search, and BLEU evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# Install: pip install torch numpy matplotlib


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """Complete Transformer model for translation"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=100):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.output_layer(x)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return output


class Vocabulary:
    """Vocabulary for source and target languages"""
    
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_count = Counter()
        self.n_words = 4
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word_count[word] += 1
    
    def sentence_to_indices(self, sentence, max_len=None):
        indices = [self.word2idx.get(word, self.word2idx['<unk>']) 
                  for word in sentence.split()]
        
        if max_len:
            if len(indices) < max_len:
                indices += [self.word2idx['<pad>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
        
        return indices
    
    def indices_to_sentence(self, indices):
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices 
                        if idx not in [self.word2idx['<pad>'], self.word2idx['<sos>'], self.word2idx['<eos>']]])


class TranslationDataset(Dataset):
    """Dataset for translation pairs"""
    
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        
        # Convert to indices
        src_indices = self.src_vocab.sentence_to_indices(src, self.max_len)
        tgt_indices = [self.tgt_vocab.word2idx['<sos>']] + \
                     self.tgt_vocab.sentence_to_indices(tgt, self.max_len - 2) + \
                     [self.tgt_vocab.word2idx['<eos>']]
        
        return torch.LongTensor(src_indices), torch.LongTensor(tgt_indices)


class TranslationTrainer:
    """Training pipeline for neural machine translation"""
    
    def __init__(self, model, src_vocab, tgt_vocab, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
    
    def create_mask(self, src, tgt):
        """Create attention masks"""
        src_mask = (src != self.src_vocab.word2idx['<pad>']).unsqueeze(1).unsqueeze(2)
        
        tgt_mask = (tgt != self.tgt_vocab.word2idx['<pad>']).unsqueeze(1).unsqueeze(3)
        
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones(1, seq_len, seq_len)).bool().to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for src, tgt in tqdm(train_loader, desc="Training"):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Prepare target (input and output)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Create masks
            src_mask, tgt_mask = self.create_mask(src, tgt_input)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc="Validation"):
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                src_mask, tgt_mask = self.create_mask(src, tgt_input)
                
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=20, lr=0.0001):
        """Complete training loop"""
        criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.word2idx['<pad>'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_translation_model.pth')
                print(f"Saved best model with loss: {best_loss:.4f}")
    
    def translate(self, sentence, max_len=50, method='greedy'):
        """Translate a sentence"""
        self.model.eval()
        
        # Prepare source
        src_indices = self.src_vocab.sentence_to_indices(sentence, max_len)
        src = torch.LongTensor(src_indices).unsqueeze(0).to(self.device)
        
        # Encode
        src_mask = (src != self.src_vocab.word2idx['<pad>']).unsqueeze(1).unsqueeze(2)
        encoder_output = self.model.encode(src, src_mask)
        
        if method == 'greedy':
            return self._greedy_decode(encoder_output, src_mask, max_len)
        else:
            return self._beam_search(encoder_output, src_mask, max_len, beam_size=5)
    
    def _greedy_decode(self, encoder_output, src_mask, max_len):
        """Greedy decoding"""
        tgt = torch.LongTensor([[self.tgt_vocab.word2idx['<sos>']]]).to(self.device)
        
        for _ in range(max_len):
            tgt_mask = torch.tril(torch.ones(1, tgt.size(1), tgt.size(1))).bool().to(self.device)
            
            output = self.model.decode(tgt, encoder_output, src_mask, tgt_mask)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            if next_token.item() == self.tgt_vocab.word2idx['<eos>']:
                break
        
        return self.tgt_vocab.indices_to_sentence(tgt[0].tolist())
    
    def _beam_search(self, encoder_output, src_mask, max_len, beam_size=5):
        """Beam search decoding"""
        sequences = [[self.tgt_vocab.word2idx['<sos>']]]
        scores = [0.0]
        
        for _ in range(max_len):
            all_candidates = []
            
            for seq, score in zip(sequences, scores):
                if seq[-1] == self.tgt_vocab.word2idx['<eos>']:
                    all_candidates.append((seq, score))
                    continue
                
                tgt = torch.LongTensor([seq]).to(self.device)
                tgt_mask = torch.tril(torch.ones(1, tgt.size(1), tgt.size(1))).bool().to(self.device)
                
                with torch.no_grad():
                    output = self.model.decode(tgt, encoder_output, src_mask, tgt_mask)
                    log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                
                top_probs, top_indices = log_probs.topk(beam_size)
                
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    candidate = (seq + [idx.item()], score + prob.item())
                    all_candidates.append(candidate)
            
            # Select top beam_size sequences
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [seq for seq, _ in ordered[:beam_size]]
            scores = [score for _, score in ordered[:beam_size]]
            
            # Check if all sequences ended
            if all(seq[-1] == self.tgt_vocab.word2idx['<eos>'] for seq in sequences):
                break
        
        best_sequence = sequences[0]
        return self.tgt_vocab.indices_to_sentence(best_sequence)
    
    def calculate_bleu(self, references, hypotheses):
        """Calculate BLEU score"""
        from collections import Counter
        
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        bleu_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            
            # Calculate precision for different n-grams
            precisions = []
            for n in range(1, 5):
                ref_ngrams = Counter(get_ngrams(ref_tokens, n))
                hyp_ngrams = Counter(get_ngrams(hyp_tokens, n))
                
                overlap = sum((ref_ngrams & hyp_ngrams).values())
                total = sum(hyp_ngrams.values())
                
                if total > 0:
                    precisions.append(overlap / total)
                else:
                    precisions.append(0)
            
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
            
            # Geometric mean
            if all(p > 0 for p in precisions):
                bleu = bp * np.exp(np.mean([np.log(p) for p in precisions]))
            else:
                bleu = 0
            
            bleu_scores.append(bleu)
        
        return np.mean(bleu_scores)
    
    def plot_training_history(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 5))
        
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Translation Training History')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('translation_training.png', dpi=300)
        plt.show()


# Generate synthetic translation data
def generate_synthetic_translation_data(n_samples=5000):
    """Generate simple synthetic translation pairs"""
    
    # Simple vocabulary
    src_words = ['i', 'you', 'he', 'she', 'we', 'love', 'like', 'eat', 'drink', 
                'apple', 'banana', 'water', 'coffee', 'book', 'car']
    tgt_words = ['je', 'tu', 'il', 'elle', 'nous', 'aime', 'aimer', 'mange', 'boire',
                'pomme', 'banane', 'eau', 'café', 'livre', 'voiture']
    
    patterns = [
        ("{subj} {verb} {obj}", "{tgt_subj} {tgt_verb} {tgt_obj}"),
        ("{subj} {verb}", "{tgt_subj} {tgt_verb}"),
        ("{obj} is good", "{tgt_obj} est bon"),
    ]
    
    src_sentences = []
    tgt_sentences = []
    
    for _ in range(n_samples):
        pattern_idx = np.random.randint(0, len(patterns))
        src_pattern, tgt_pattern = patterns[pattern_idx]
        
        if '{subj}' in src_pattern:
            subj_idx = np.random.randint(0, 5)
            src_pattern = src_pattern.replace('{subj}', src_words[subj_idx])
            tgt_pattern = tgt_pattern.replace('{tgt_subj}', tgt_words[subj_idx])
        
        if '{verb}' in src_pattern:
            verb_idx = np.random.randint(5, 9)
            src_pattern = src_pattern.replace('{verb}', src_words[verb_idx])
            tgt_pattern = tgt_pattern.replace('{tgt_verb}', tgt_words[verb_idx])
        
        if '{obj}' in src_pattern:
            obj_idx = np.random.randint(9, 15)
            src_pattern = src_pattern.replace('{obj}', src_words[obj_idx])
            tgt_pattern = tgt_pattern.replace('{tgt_obj}', tgt_words[obj_idx])
        
        src_sentences.append(src_pattern)
        tgt_sentences.append(tgt_pattern)
    
    return src_sentences, tgt_sentences


if __name__ == "__main__":
    print("Generating synthetic translation data...")
    src_sentences, tgt_sentences = generate_synthetic_translation_data(n_samples=3000)
    
    # Build vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    for src, tgt in zip(src_sentences, tgt_sentences):
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)
    
    print(f"Source vocabulary size: {src_vocab.n_words}")
    print(f"Target vocabulary size: {tgt_vocab.n_words}")
    
    # Split data
    split_idx = int(0.8 * len(src_sentences))
    train_src, val_src = src_sentences[:split_idx], src_sentences[split_idx:]
    train_tgt, val_tgt = tgt_sentences[:split_idx], tgt_sentences[split_idx:]
    
    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("\nInitializing Transformer model...")
    model = Transformer(
        src_vocab_size=src_vocab.n_words,
        tgt_vocab_size=tgt_vocab.n_words,
        d_model=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1
    )
    
    # Train
    trainer = TranslationTrainer(model, src_vocab, tgt_vocab)
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, epochs=20, lr=0.0001)
    
    # Plot
    trainer.plot_training_history()
    
    # Test translations
    print("\n" + "="*50)
    print("SAMPLE TRANSLATIONS")
    print("="*50)
    
    test_sentences = val_src[:5]
    for src in test_sentences:
        greedy = trainer.translate(src, method='greedy')
        beam = trainer.translate(src, method='beam_search')
        print(f"\nSource: {src}")
        print(f"Greedy: {greedy}")
        print(f"Beam:   {beam}")
    
    print("\nTraining complete! Model saved as 'best_translation_model.pth'")