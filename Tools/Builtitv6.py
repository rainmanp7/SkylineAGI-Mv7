import subprocess
import os
from multiprocessing import Pool
import json
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from logging_config import setup_logging


def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return content, os.path.basename(file_path)
    except Exception as e:
        logger.error(f'Error reading file {file_path}: {e}')
        return None, None


def rate_interestingness(content):
    ml_keywords = ['import torch', 'nn', 'optim', 'layer', 'forward', 'backward', 'loss', 'data', 'train', 'val']
    return sum(content.lower().count(keyword) for keyword in ml_keywords) + content.count('\n')


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = input_size
        self.d_k = input_size // num_heads
        self.d_v = input_size // num_heads
        self.query_layer = nn.Linear(input_size, input_size)
        self.key_layer = nn.Linear(input_size, input_size)
        self.value_layer = nn.Linear(input_size, input_size)
        self.attention_combine = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.size(0)
        queries = self.query_layer(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        keys = self.key_layer(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        values = self.value_layer(x).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.attention_combine(context)
        output = self.dropout(output)
        return output


class SimpleModel(nn.Module):
    def __init__(self, input_size, num_heads, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(input_size, num_heads, dropout_rate)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.attention(x)
        x = x.mean(dim=1)  # Ensure proper dimension reduction
        x = self.fc(x)
        return x


def load_data():
    X_train = torch.randn(100, 10, 128)
    y_train = torch.randn(100, 1)
    X_val = torch.randn(20, 10, 128)
    y_val = torch.randn(20, 1)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    return train_loader, val_loader


def objective(trial, model, train_loader, val_loader):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(5):  # Short training for optimization
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = sum(criterion(model(batch_x), batch_y).item() for batch_x, batch_y in val_loader) / len(val_loader)
    return val_loss


def collect_code(root_dir):
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(root_dir) for file in files if file.endswith(('.py', '.txt'))]
    with Pool() as pool:
        results = pool.map(process_file, file_paths)
    return [code for code, _ in results if code], [path for _, path in results if path]


def generate_code(code_snippet, target_file):
    with open(target_file, 'w') as f:
        f.write(code_snippet + '\n')  # Ensure newline for Pylint
    logger.info(f'Code snippet generated and saved to {target_file}')


def optimize_code(target_file):
    try:
        result = subprocess.run(['pylint', target_file], capture_output=True, text=True)
        logger.info(f'Pylint output:\n{result.stdout}')
        return result.stdout
    except Exception as e:
        logger.error(f'Error running pylint: {e}')
        return None


def build_and_optimize_model(model_config, train_loader, val_loader, num_rounds, code_gen_config):
    input_size = model_config['input_size']
    num_heads = model_config['num_heads']
    dropout_rate = model_config['dropout_rate']
    code_generation_enabled = code_gen_config['enabled']
    target_file = code_gen_config['target_file']
    initial_avg_val_loss = None

    for round in range(1, num_rounds + 1):
        logger.info(f'### Round {round} ###')
        model = SimpleModel(input_size, num_heads, dropout_rate)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, model, train_loader, val_loader), n_trials=2)
        best_params = study.best_params
        logger.info(f'Best parameters: {best_params}')
        logger.info(f'Best validation loss: {study.best_value}')

        if code_generation_enabled:
            code_snippet = f"# Optimized model code for round {round}."
            generate_code(code_snippet, target_file)
            optimize_code(target_file)


def extract_and_save_interesting_snippets(code_snippets, file_names, target_snippet_file):
    scores = [rate_interestingness(snippet) for snippet in code_snippets]
    sorted_snippets = sorted(zip(scores, code_snippets, file_names), reverse=True)
    with open(target_snippet_file, 'w') as f:
        for score, snippet, file_name in sorted_snippets:
            f.write(f"### {file_name} ({score})\n{snippet}\n\n")
    logger.info(f'Snippets saved to {target_snippet_file}')


logger = setup_logging(script_name='ai_model_builder')

if __name__ == '__main__':
    num_rounds = int(input('Enter the number of rounds: '))
    logger.info(f'Number of rounds: {num_rounds}')
    with open('config.json', 'r') as f:
        config = json.load(f)
    model_config = config['model_config']
    code_gen_config = config['code_generation']
    train_loader, val_loader = load_data()
    code_snippets, file_names = collect_code('.')
    extract_and_save_interesting_snippets(code_snippets, file_names, 'interesting_snippets.txt')
    build_and_optimize_model(model_config, train_loader, val_loader, num_rounds, code_gen_config)
