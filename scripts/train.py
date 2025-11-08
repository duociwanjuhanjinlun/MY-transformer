import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
from tqdm import tqdm
from src.models.transformer import Transformer

def train_epoch(model, dataloader, criterion, optimizer, pad_idx):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 创建mask
        src_mask = model.create_pad_mask(src, pad_idx)
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        
        # 计算损失
        loss = criterion(output.contiguous().view(-1, output.size(-1)),
                       tgt[:, 1:].contiguous().view(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, pad_idx):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
            src_mask = model.create_pad_mask(src, pad_idx)
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = criterion(output.contiguous().view(-1, output.size(-1)),
                           tgt[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型参数
    src_vocab_size = 10000  # 根据实际数据集调整
    tgt_vocab_size = 10000  # 根据实际数据集调整
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    
    # 训练参数
    num_epochs = 10
    learning_rate = 0.0001
    batch_size = 32
    pad_idx = 0
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # 训练循环
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, pad_idx)
        valid_loss = validate(model, valid_dataloader, criterion, pad_idx)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch: {epoch+1}")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tValid Loss: {valid_loss:.4f}")
        print(f"\tEpoch time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')