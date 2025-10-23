
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import time
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir  = "results_run"
os.makedirs(base_dir, exist_ok=True)
run_id = len(os.listdir(    base_dir))+1
run_folder = os.path.join(base_dir, f"run_{run_id:03d}")
os.makedirs(run_folder, exist_ok=True)
CSV_PATH = os.path.join(run_folder, f"run_{run_id:03d}.csv")


train_data = np.loadtxt('1kPCA.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('1kPCA.csv', delimiter=',', skiprows=1)

print("Train unique labels:", np.unique(train_data[:, -1]))
print("Test  unique labels:", np.unique(test_data[:, -1]))


x_train = torch.tensor(train_data[:, :-1], dtype=torch.float32).to(device)
y_train = torch.tensor(train_data[:,-1], dtype=torch.float32).unsqueeze(1).to(device)

X_test  = torch.tensor(test_data[:, :-1], dtype=torch.float32).to(device)
y_test  = torch.tensor(test_data[:, -1],  dtype=torch.float32).unsqueeze(1).to(device)
# .unsqueeze(1)






class ANFIS(nn.Module):
    def __init__(self, n_inputs: int, n_rules: int = 3):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        # Premise: Gaussian MF params
        self.mean  = nn.Parameter(torch.randn(n_inputs, n_rules))
        self.sigma = nn.Parameter(torch.abs(torch.randn(n_inputs, n_rules)) + 0.1)
        # Consequent: [bias + n_inputs] per rule (به‌روزرسانی با LSE)
        self.conseq = nn.Parameter(torch.zeros(n_rules, n_inputs + 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1: MF values
        mf = torch.exp(-((x.unsqueeze(2) - self.mean)**2) / (2 * self.sigma**2))  # [B, n_inputs, n_rules]
        # Layer 2: rule firing strength (product across inputs)
        w = torch.prod(mf, dim=1)  # [B, n_rules]
        # Layer 3: normalization
        w_norm = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)
        # Layer 4: linear consequents f_i(x) = p_i^T [1,x]
        x_ext = torch.cat([torch.ones(x.size(0), 1, device=x.device), x], dim=1)  # [B, n_inputs+1]
        rule_out = torch.matmul(x_ext, self.conseq.T)  # [B, n_rules]
        # Layer 5: weighted sum
        y = torch.sum(w_norm * rule_out, dim=1, keepdim=True)  # [B,1]
        return y

@torch.no_grad()
def lse_update(model: ANFIS, x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-6) -> None:
    """
    به‌روزرسانی بستهٔ ضرایب پیامد (conseq) با LSE.
    A θ = y  ، که در آن A از w_norm و [1,x] ساخته می‌شود.
    """
    device = x.device
    mf = torch.exp(-((x.unsqueeze(2) - model.mean)**2) / (2 * model.sigma**2))
    w = torch.prod(mf, dim=1)
    w_norm = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)  # [B, R]

    x_ext = torch.cat([torch.ones(x.size(0), 1, device=device), x], dim=1)  # [B, d+1]
    B, R, Dp1 = x.size(0), model.n_rules, model.n_inputs + 1

    # Design matrix A: block columns per rule: w_norm[:,j] * x_ext
    A = torch.zeros(B, R * Dp1, device=device)
    for j in range(R):
        A[:, j*Dp1:(j+1)*Dp1] = w_norm[:, j].unsqueeze(1) * x_ext

    ATA = A.T @ A + ridge * torch.eye(R * Dp1, device=device)
    ATy = A.T @ y
    theta = torch.linalg.solve(ATA, ATy).reshape(R, Dp1)
    model.conseq.copy_(theta)


# def train_hybrid(model: ANFIS, x: torch.Tensor, y: torch.Tensor, epochs: int = 50, lr: float = 1e-3):
def train_hybrid(model, x, y, optimizer, criterion):
    lse_update(model, x, y)
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        model.sigma.clamp_(min=1e-3)
    return loss
    # """
    # آموزش ترکیبی: در هر ایپوک ابتدا LSE برای consequent سپس GD برای premise.
    # """
    # # optimizer = optim.Adam([model.mean, model.sigma], lr=lr)
    # optimizer = torch.optim.Adam([model.mean, model.sigma], lr=lr)
    # loss_fn = nn.MSELoss()
    # for ep in range(epochs):
    #     # Phase 1: LSE (closed-form)
    #     lse_update(model, x, y)
    #     # Phase 2: GD for premise params
    #     optimizer.zero_grad()
    #     y_hat = model(x)
    #     loss = loss_fn(y_hat, y)
    #     loss.backward()
    #     optimizer.step()
    #     # optional: clamp sigma for numerical stability
    #     with torch.no_grad():
    #         model.sigma.clamp_(min=1e-3)
    #     if (ep+1) % max(1, epochs//5) == 0:
    #         print(f"[Hybrid] Epoch {ep+1}/{epochs}  RMSE={torch.sqrt(loss).item():.6f}")
    # return model

@torch.no_grad()
def eval_metrics(y_true: torch.Tensor, y_pred_cont: torch.Tensor) -> dict:
    """
    محاسبهٔ متریک‌ها: RMSE/MSE/MAE و برای طبقه‌بندی با سیگموید و آستانه تطبیقی.
    """
    # Regression metrics
    e = (y_true - y_pred_cont)
    mse = torch.mean(e**2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mae = torch.mean(torch.abs(e)).item()

    # Binary metrics with adaptive threshold
    prob = torch.sigmoid(y_pred_cont)
    thr = torch.quantile(prob, 0.5)
    y_bin = (prob > thr).int()
    t_bin = (y_true > 0).int()
    tp = int(((y_bin == 1) & (t_bin == 1)).sum().item())
    fp = int(((y_bin == 1) & (t_bin == 0)).sum().item())
    fn = int(((y_bin == 0) & (t_bin == 1)).sum().item())
    tn = int(((y_bin == 0) & (t_bin == 0)).sum().item())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2*precision*recall/(precision+recall)

    return {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Threshold": float(thr.item())
    }

 

model = ANFIS(n_inputs=x_train.shape[1], n_rules=3).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([227450 / 394], device=device))
μ, β = 0.01, 0.2
asdsadsad = 11
optimizer = torch.optim.SGD(model.parameters(), lr=μ)


epochs = 20
records = []
e1 = float('inf')
start_time = time.time()

# ==== حلقه آموزش ====s
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    e2 = loss.item()

    # Adaptive μ
    if e2 >= e1:
        μ *= β
    else:
        μ /= β
    μ = float(np.clip(μ, 1e-5, 0.1))
    for g in optimizer.param_groups: g["lr"] = μ

    loss.backward()
    # optimizer.step()
    loss = train_hybrid(model, x_train, y_train, optimizer, criterion)
    
    e1 = e2

# ==== ارزیابی ====
model.eval()
with torch.no_grad():
    out_test = model(X_test)
    probs = torch.sigmoid(out_test).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_true = y_test.cpu().numpy()
    y_true = (y_true > 0.5).astype(int)
    preds = (preds > 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    rmse = np.sqrt(((preds - y_true) ** 2).mean())
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

records.append({
    "epoch": epoch+1, "loss": e2, "mu": μ,
    "accuracy": acc, "rmse": rmse, "precision": precision,
    "recall": recall, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc
})

print(f"Epoch {epoch+1:03d} | Loss={e2:.6f} | ACC={acc:.4f} | μ={μ:.6f}")
print(cm)

end_time = time.time()

# ==== ذخیره نتایج ====
df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)

cm_expanded = np.vstack([
    ['', 'Pred_Neg', 'Pred_Pos'],
    ['True_Neg', cm[0,0], cm[0,1]],
    ['True_Pos', cm[1,0], cm[1,1]],
    ['', '', ''],
    ['Run_Time_sec', end_time-start_time, '']
])
pd.DataFrame(cm_expanded).to_csv(os.path.join(run_folder, f"confusion_run_{run_id:03d}.csv"), index=False)

# ==== نمودار ====

plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["accuracy"], marker='o', label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Run {run_id} - Accuracy per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"accuracy_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"Accuracy plot saved at: {IMG_PATH}")

#pr_auc
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["pr_auc"], marker='o', label="pr_auc")
plt.xlabel("Epoch")
plt.ylabel("pr_auc")
plt.title(f"Run {run_id} - pr_auc per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"pr_auc_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"pr_auc plot saved at: {IMG_PATH}")

#roc_auc
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["roc_auc"], marker='o', label="roc_auc")
plt.xlabel("Epoch")
plt.ylabel("roc_auc")
plt.title(f"Run {run_id} - roc_auc per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"roc_auc_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"roc_auc plot saved at: {IMG_PATH}")


#f1
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["f1"], marker='o', label="f1")
plt.xlabel("Epoch")
plt.ylabel("f1")
plt.title(f"Run {run_id} - f1 per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"f1_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"f1 plot saved at: {IMG_PATH}")




print(f"\nTraining finished in {end_time-start_time:.2f}s")
print(f"Results saved in: {run_folder}")

 

 
