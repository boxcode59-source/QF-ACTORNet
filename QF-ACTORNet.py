# ============================================================
# QF-ACTORNet
# Quantum Convolutional Deep Learning with Actor Optimization
# for Intelligent Financial Market Prediction
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import pywt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel

# ============================================================
# PHASE 1 : DATA ACQUISITION
# ============================================================

def download_stock_data(symbol="AAPL",
                        start="2018-01-01",
                        end="2025-01-01"):

    data = yf.download(symbol,
                       start=start,
                       end=end)

    return data


# ============================================================
# PHASE 2 : AFNF
# Adaptive Financial Noise Filtration
# ============================================================

class AFNF:

    def wavelet_denoising(self, signal):

        coeffs = pywt.wavedec(signal,
                              'db4',
                              level=3)

        threshold = np.std(coeffs[-1]) * np.sqrt(
            2*np.log(len(signal))
        )

        coeffs[1:] = [
            pywt.threshold(c,
                           threshold,
                           mode='soft')
            for c in coeffs[1:]
        ]

        return pywt.waverec(coeffs,
                            'db4')

    def remove_outliers(self, df):

        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)

        IQR = Q3 - Q1

        return df[
            ~((df < (Q1 - 1.5 * IQR)) |
              (df > (Q3 + 1.5 * IQR))).any(axis=1)
        ]

    def normalize(self, data):

        scaler = MinMaxScaler()

        return scaler.fit_transform(data)


# ============================================================
# PHASE 3 : MMSE
# Multi-Horizon Market Sequence Encoder
# ============================================================

class MMSE(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim):

        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attn = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        attn_out, _ = self.attn(
            out,
            out,
            out
        )

        return attn_out


# ============================================================
# FINANCIAL SENTIMENT ENCODING
# ============================================================

class FinancialSentimentEncoder:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert"
        )

        self.model = AutoModel.from_pretrained(
            "ProsusAI/finbert"
        )

    def encode(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            output = self.model(**inputs)

        return output.last_hidden_state.mean(
            dim=1
        )


# ============================================================
# PHASE 4 : QCDL-NET
# Quantum Convolutional Deep Learning
# ============================================================

class QuantumConvLayer(nn.Module):

    def __init__(self,
                 in_features,
                 out_features):

        super().__init__()

        self.fc = nn.Linear(
            in_features,
            out_features
        )

    def forward(self, x):

        x = torch.sin(self.fc(x))
        return x


class QuantumAttention(nn.Module):

    def __init__(self,
                 dim):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x):

        out, _ = self.attention(
            x,
            x,
            x
        )

        return out


class QCDLNet(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim):

        super().__init__()

        self.qconv1 = QuantumConvLayer(
            input_dim,
            hidden_dim
        )

        self.qconv2 = QuantumConvLayer(
            hidden_dim,
            hidden_dim
        )

        self.qattn = QuantumAttention(
            hidden_dim
        )

    def forward(self, x):

        x = self.qconv1(x)

        x = self.qconv2(x)

        x = self.qattn(x)

        return x


# ============================================================
# PHASE 5 : ACTOR OPTIMIZATION
# ============================================================

class ActorNetwork(nn.Module):

    def __init__(self,
                 state_dim,
                 action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,state):

        return self.net(state)


class ActorOptimizer:

    def __init__(self,
                 state_dim,
                 action_dim):

        self.actor = ActorNetwork(
            state_dim,
            action_dim
        )

        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=1e-4
        )

    def update(self,
               state,
               reward):

        probs = self.actor(state)

        loss = -reward * torch.mean(
            torch.log(probs)
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


# ============================================================
# PHASE 6 : TRUST-AWARE FINANCIAL DECISION FUSION
# ============================================================

class TFDF(nn.Module):

    def __init__(self,
                 dim):

        super().__init__()

        self.confidence_layer = nn.Linear(
            dim,
            1
        )

    def forward(self, x):

        confidence = torch.sigmoid(
            self.confidence_layer(x)
        )

        trusted_output = x * confidence

        return trusted_output, confidence


# ============================================================
# PHASE 7 : PREDICTION NETWORK
# ============================================================

class MarketPredictor(nn.Module):

    def __init__(self,
                 input_dim):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(input_dim,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,3)
        )

    def forward(self,x):

        return self.fc(x)


# ============================================================
# PHASE 8 : EXPLAINABLE TRADING RECOMMENDATION
# ============================================================

def trading_recommendation(
        prediction,
        confidence):

    label = np.argmax(prediction)

    if label == 0:
        action = "SELL"

    elif label == 1:
        action = "HOLD"

    else:
        action = "BUY"

    return {
        "Recommendation": action,
        "Confidence": float(confidence)
    }


# ============================================================
# COMPLETE QF-ACTORNET
# ============================================================

class QF_ACTORNet(nn.Module):

    def __init__(self,
                 input_dim):

        super().__init__()

        self.mmse = MMSE(
            input_dim=input_dim,
            hidden_dim=128
        )

        self.qcdl = QCDLNet(
            input_dim=256,
            hidden_dim=128
        )

        self.tfdf = TFDF(128)

        self.predictor = MarketPredictor(
            128
        )

    def forward(self,x):

        x = self.mmse(x)

        x = self.qcdl(x)

        x = x.mean(dim=1)

        trusted,
        confidence = self.tfdf(x)

        pred = self.predictor(trusted)

        return pred, confidence


# ============================================================
# TRAINING
# ============================================================

def train_model(model,
                train_loader,
                epochs=50):

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for x,y in train_loader:

            pred, conf = model(x)

            loss = criterion(pred,y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1} "
            f"Loss:{total_loss:.4f}"
        )

    return model


# ============================================================
# INFERENCE
# ============================================================

def predict(model,x):

    model.eval()

    with torch.no_grad():

        pred,
        confidence = model(x)

        pred = torch.softmax(
            pred,
            dim=1
        )

    return trading_recommendation(
        pred[0].cpu().numpy(),
        confidence.mean().item()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    stock_data = download_stock_data(
        "AAPL"
    )

    print(stock_data.head())

    model = QF_ACTORNet(
        input_dim=20
    )

    print(model)