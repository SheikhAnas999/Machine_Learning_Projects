"""
Question-Answering System using BERT
Fine-tuned for domain-specific QA with extractive answers
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import json

# Install: pip install torch transformers


class QADataset(Dataset):
    """Dataset for Question Answering"""
    
    def __init__(self, contexts, questions, answers, tokenizer, max_length=384):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Find answer positions
        answer_text = answer['text']
        start_char = answer['answer_start']
        end_char = start_char + len(answer_text)
        
        # Convert character positions to token positions
        start_token = 0
        end_token = 0
        
        # Get token to character mapping
        token_start_index = 0
        token_end_index = len(encoding['input_ids'][0]) - 1
        
        # Find start position
        for i in range(token_start_index, token_end_index + 1):
            span = encoding.token_to_chars(0, i)
            if span is not None and span.start <= start_char < span.end:
                start_token = i
                break
        
        # Find end position
        for i in range(token_end_index, token_start_index - 1, -1):
            span = encoding.token_to_chars(0, i)
            if span is not None and span.start < end_char <= span.end:
                end_token = i
                break
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_token),
            'end_positions': torch.tensor(end_token)
        }


class QuestionAnsweringSystem:
    """Complete QA system with training and inference"""
    
    def __init__(self, model_name='bert-base-uncased', 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name).to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.exact_match_scores = []
        self.f1_scores = []
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, contexts, questions, answers):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate exact match and F1 scores on subset
        em_score, f1_score = self.evaluate_predictions(
            contexts[:100], questions[:100], answers[:100]
        )
        
        return avg_loss, em_score, f1_score
    
    def train(self, train_loader, val_loader, val_contexts, val_questions, 
              val_answers, epochs=3, lr=3e-5):
        """Complete training loop"""
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        best_f1 = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_loss, em_score, f1_score = self.validate(
                val_loader, val_contexts, val_questions, val_answers
            )
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.exact_match_scores.append(em_score)
            self.f1_scores.append(f1_score)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Exact Match: {em_score:.2f}%")
            print(f"F1 Score: {f1_score:.2f}%")
            
            # Save best model
            if f1_score > best_f1:
                best_f1 = f1_score
                self.model.save_pretrained('best_qa_model')
                self.tokenizer.save_pretrained('best_qa_model')
                print(f"Saved best model with F1: {best_f1:.2f}%")
    
    def answer_question(self, context, question, return_score=False):
        """Answer a single question"""
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            max_length=384,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get answer span
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        
        # Get confidence score
        start_score = torch.max(torch.softmax(start_logits, dim=1)).item()
        end_score = torch.max(torch.softmax(end_logits, dim=1)).item()
        confidence = (start_score + end_score) / 2
        
        # Decode answer
        answer_tokens = input_ids[0][start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        if return_score:
            return answer, confidence
        return answer
    
    def evaluate_predictions(self, contexts, questions, answers):
        """Evaluate predictions using Exact Match and F1"""
        exact_matches = 0
        f1_total = 0
        
        for context, question, answer in zip(contexts, questions, answers):
            predicted = self.answer_question(context, question)
            reference = answer['text'].lower().strip()
            predicted = predicted.lower().strip()
            
            # Exact match
            if predicted == reference:
                exact_matches += 1
            
            # F1 score
            pred_tokens = predicted.split()
            ref_tokens = reference.split()
            
            common = set(pred_tokens) & set(ref_tokens)
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1 = 0
            elif len(common) == 0:
                f1 = 0
            else:
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(ref_tokens)
                f1 = 2 * (precision * recall) / (precision + recall)
            
            f1_total += f1
        
        n = len(questions)
        exact_match = 100.0 * exact_matches / n
        f1_score = 100.0 * f1_total / n
        
        return exact_match, f1_score
    
    def batch_answer(self, contexts, questions):
        """Answer multiple questions efficiently"""
        answers = []
        confidences = []
        
        for context, question in zip(contexts, questions):
            answer, confidence = self.answer_question(
                context, question, return_score=True
            )
            answers.append(answer)
            confidences.append(confidence)
        
        return answers, confidences


# Generate synthetic QA data
def generate_synthetic_qa_data(n_samples=500):
    """Generate synthetic question-answering pairs"""
    contexts = []
    questions = []
    answers = []
    
    templates = [
        {
            'context': "The {animal} is a {adjective} creature that lives in {location}. "
                      "It primarily eats {food} and can grow up to {size} meters long.",
            'questions': [
                "Where does the {animal} live?",
                "What does the {animal} eat?",
                "How long can the {animal} grow?"
            ],
            'answer_keys': ['{location}', '{food}', '{size} meters']
        },
        {
            'context': "In {year}, {person} invented the {invention}. "
                      "This revolutionary device changed {field} forever and cost ${cost} to produce.",
            'questions': [
                "Who invented the {invention}?",
                "When was the {invention} invented?",
                "How much did it cost to produce?"
            ],
            'answer_keys': ['{person}', '{year}', '${cost}']
        }
    ]
    
    animals = ['elephant', 'tiger', 'dolphin', 'eagle', 'python']
    adjectives = ['magnificent', 'powerful', 'intelligent', 'graceful', 'swift']
    locations = ['Africa', 'Asia', 'ocean', 'mountains', 'jungle']
    foods = ['grass', 'meat', 'fish', 'small animals', 'insects']
    sizes = ['3', '2.5', '10', '1.5', '5']
    
    people = ['John Smith', 'Marie Curie', 'Thomas Edison', 'Ada Lovelace', 'Einstein']
    years = ['1920', '1935', '1950', '1965', '1980']
    inventions = ['computer', 'telephone', 'radio', 'calculator', 'microscope']
    fields = ['technology', 'communication', 'science', 'medicine', 'education']
    costs = ['1000', '5000', '500', '2000', '3000']
    
    values_list = [
        {'animal': animals, 'adjective': adjectives, 'location': locations,
         'food': foods, 'size': sizes},
        {'person': people, 'year': years, 'invention': inventions,
         'field': fields, 'cost': costs}
    ]
    
    for _ in range(n_samples):
        template_idx = np.random.randint(0, len(templates))
        template = templates[template_idx]
        values = values_list[template_idx]
        
        # Fill template
        filled_context = template['context']
        for key, options in values.items():
            value = np.random.choice(options)
            filled_context = filled_context.replace('{' + key + '}', value, 1)
        
        # Choose random question
        q_idx = np.random.randint(0, len(template['questions']))
        filled_question = template['questions'][q_idx]
        answer_text = template['answer_keys'][q_idx]
        
        for key, options in values.items():
            value = np.random.choice(options)
            filled_question = filled_question.replace('{' + key + '}', value, 1)
            answer_text = answer_text.replace('{' + key + '}', value, 1)
        
        # Find answer position
        answer_start = filled_context.find(answer_text)
        
        if answer_start != -1:
            contexts.append(filled_context)
            questions.append(filled_question)
            answers.append({
                'text': answer_text,
                'answer_start': answer_start
            })
    
    return contexts, questions, answers


if __name__ == "__main__":
    print("Generating synthetic QA data...")
    contexts, questions, answers = generate_synthetic_qa_data(n_samples=1000)
    
    # Split data
    split_idx = int(0.8 * len(contexts))
    train_contexts = contexts[:split_idx]
    train_questions = questions[:split_idx]
    train_answers = answers[:split_idx]
    
    val_contexts = contexts[split_idx:]
    val_questions = questions[split_idx:]
    val_answers = answers[split_idx:]
    
    # Initialize system
    print("\nInitializing QA system...")
    qa_system = QuestionAnsweringSystem()
    
    # Create datasets
    train_dataset = QADataset(
        train_contexts, train_questions, train_answers, qa_system.tokenizer
    )
    val_dataset = QADataset(
        val_contexts, val_questions, val_answers, qa_system.tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Train
    print("\nStarting training...")
    qa_system.train(
        train_loader, val_loader, val_contexts, val_questions, val_answers,
        epochs=3, lr=3e-5
    )
    
    # Test on examples
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    for i in range(5):
        context = val_contexts[i]
        question = val_questions[i]
        true_answer = val_answers[i]['text']
        
        predicted_answer, confidence = qa_system.answer_question(
            context, question, return_score=True
        )
        
        print(f"\nContext: {context[:150]}...")
        print(f"Question: {question}")
        print(f"True Answer: {true_answer}")
        print(f"Predicted: {predicted_answer}")
        print(f"Confidence: {confidence:.2f}")
    
    print("\nTraining complete! Model saved to 'best_qa_model/'")