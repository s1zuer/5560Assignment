import torch
import torch.nn as nn
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = get_data_loader("./data", batch_size=64, train=True)
test_loader  = get_data_loader("./data", batch_size=64, train=False)

model = get_model("CNN")           # 或 "EnhancedCNN"
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=5)
test_loss, test_acc = evaluate_model(model, test_loader, criterion, device=device)
print(f"Test loss={test_loss:.4f} acc={test_acc:.4f}")

save_model(model, "./artifacts/cnn_best.pt")

from helper_lib.visualize_results import visualize_predictions

# …前面的训练代码不变…
save_model(model, path="./artifacts/cnn_best.pt")

# 显示预测结果
visualize_predictions(model_path="./artifacts/cnn_best.pt")

if __name__ == "__main__":
    # 这里放你现在的训练 + 评估 + save_model + 可视化
    pass