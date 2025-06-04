import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Data prep (exactly as before) ────────────────────────────
data = pd.read_csv('CourseraDataset.csv')
X_full = data.drop(columns=['CustomerID', 'Churn'])
X_full = pd.get_dummies(X_full).astype(np.float32)
y_full = data['Churn'].astype(np.float32)

X_train_df, X_test_df, y_train_ser, y_test_ser = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# Convert to float32 tensors
train_inputs = torch.tensor(X_train_df.values)                # [4 × num_feats]
train_labels = torch.tensor(y_train_ser.values).unsqueeze(1)  # [4 × 1]

test_inputs  = torch.tensor(X_test_df.values)                 # [1 × num_feats]
test_labels  = torch.tensor(y_test_ser.values).unsqueeze(1)   # [1 × 1]

# ─── Simple Logistic Model ────────────────────────────────────
class SimpleLogistic(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        # Return a raw “logit” (no sigmoid here)
        return self.linear(x)

num_feats = X_train_df.shape[1]
model = SimpleLogistic(num_feats)

# ─── Loss (logits) + optimizer ───────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ─── Training loop (20 epochs) ───────────────────────────────
for epoch in range(20):
    model.train()
    optimizer.zero_grad()

    logits = model(train_inputs)                   # shape [4,1]
    loss   = criterion(logits, train_labels)       # BCEWithLogitsLoss
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}/20   Loss: {loss.item():.4f}")

# ─── Evaluation on test row ──────────────────────────────────
model.eval()
with torch.no_grad():
    test_logits = model(test_inputs)                     # raw logit
    test_prob   = torch.sigmoid(test_logits).item()      # convert to probability
    test_pred   = int(test_prob > 0.5)                   # threshold at 0.5
    test_actual = int(y_test_ser.values[0])

    print()
    print(f"Test row raw logit: {test_logits.item():.4f}")
    print(f"Test row probability: {test_prob:.4f}")
    print(f"Thresholded prediction: {test_pred}")
    print(f"Ground‐truth label   : {test_actual}")
    print(f"Test accuracy (1 sample): {int(test_pred == test_actual)}")


# Simpler Model was reequired due to limited data the examples were red herrings
# Neural Networks need 1000's of examples.