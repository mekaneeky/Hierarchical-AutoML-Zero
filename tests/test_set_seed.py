
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from automl.utils import set_seed

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def create_random_data():
    return torch.randn(100, 10), torch.randn(100, 1)

def train_and_evaluate(seed):
    set_seed(seed)
    
    # Create random data
    X, y = create_random_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, 
                            generator=torch.Generator().manual_seed(seed))

    # Create and train model
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(5):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_X, test_y = create_random_data()
        test_output = model(test_X)
        test_loss = criterion(test_output, test_y)

    return test_loss.item()

def run_tests():
    # Test 1: Same seed should produce same results
    seed = 42
    result1 = train_and_evaluate(seed)
    result2 = train_and_evaluate(seed)
    assert np.isclose(result1, result2), f"Same seed produced different results: {result1} vs {result2}"
    print("Test 1 passed: Same seed produces consistent results.")

    # Test 2: Different seeds should produce different results
    seed1, seed2 = 42, 24
    result1 = train_and_evaluate(seed1)
    result2 = train_and_evaluate(seed2)
    assert not np.isclose(result1, result2), f"Different seeds produced same result: {result1}"
    print("Test 2 passed: Different seeds produce different results.")

    # Test 3: Numpy random generation
    set_seed(42)
    np_result1 = np.random.rand()
    set_seed(42)
    np_result2 = np.random.rand()
    assert np_result1 == np_result2, "Numpy random generation not consistent with same seed."
    print("Test 3 passed: Numpy random generation is consistent with same seed.")

    # Test 4: Python's random module
    set_seed(42)
    py_result1 = random.random()
    set_seed(42)
    py_result2 = random.random()
    assert py_result1 == py_result2, "Python's random module not consistent with same seed."
    print("Test 4 passed: Python's random module is consistent with same seed.")

if __name__ == "__main__":
    run_tests()
    print("All tests passed!")