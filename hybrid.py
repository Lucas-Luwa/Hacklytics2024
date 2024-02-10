from simplemodelv2 import model
from LSTMPredictor import lstm
import numpy as np

result, pred1, accuracy1, accuracy2 = lstm()
result2, pred2, accuracy = model()
# print(result)
# print(result2)

# print("XXXXX")
# print(pred1)
# print(pred2)

merged = [a & b for a, b in zip(result, result2)]
# print(result)
# print(result2)
# print(merged)
# print(pred2)
equalOrZero = np.logical_or(np.array(merged) == pred2, np.array(merged) == 0)
# print("HEYA")
# print(equalOrZero)
convToInt = equalOrZero.astype(int)
highAccuracy = np.mean(convToInt)
normAccuracy = np.mean(np.array(merged) == pred2)

print(highAccuracy, normAccuracy)

