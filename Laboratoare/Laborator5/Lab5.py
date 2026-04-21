import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
training_data = np.load('training_data.npy')
prices = np.load('prices.npy')
# print('The first 4 samples are:\n ', training_data[:4])
# print('The first 4 prices are:\n ', prices[:4])
# training_data, prices = shuffle(training_data, prices, random_state=0)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

#1
def normalizeData(train_data, test_data):
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_data)
    test_normalized = scaler.fit_transform(test_data)
    return train_normalized, test_normalized

#2
num_samples = len(training_data)
fold_size = num_samples // 3
mse = []
mae =[]

for i in range (3):
    start = i * fold_size
    end = (i + 1) * fold_size

    X_test = training_data[start:end]
    y_test = prices[start:end]

    X_train = np.concatenate([training_data[:start], training_data[end:]], axis=0)
    y_train = np.concatenate([prices[:start], prices[end:]], axis=0)

    X_train_norm, X_test_norm = normalizeData(X_train, X_test)

    model = LinearRegression()
    model.fit(X_train_norm, y_train)

    y_pred = model.predict(X_test_norm)

    mse.append(mean_squared_error(y_test, y_pred))
    mae.append(mean_absolute_error(y_test, y_pred))

    print(np.mean(mse))
    print(np.mean(mae))


print("------------------------------------------------")
#3
alphas = [1, 10, 100, 1000]
num_samples = len(training_data)
fold_size = num_samples // 3
results = {}

for a in alphas:
    mse_fold_results = []
    mae_fold_results = []
    for i in range(3):
        start, end = i * fold_size, (i + 1) * fold_size
        X_test = training_data[start:end]
        y_test = prices[start:end]
        X_train = np.concatenate([training_data[:start], training_data[end:]], axis=0)
        y_train = np.concatenate([prices[:start], prices[end:]], axis=0)

        X_train_norm, X_test_norm = normalizeData(X_train, X_test)

        ridge_model = Ridge(alpha=a)
        ridge_model.fit(X_train_norm, y_train)

        y_pred = ridge_model.predict(X_test_norm)
        mse_fold_results.append(mean_squared_error(y_test, y_pred))
        mae_fold_results.append(mean_absolute_error(y_test, y_pred))

    print(f"mse {np.mean(mse_fold_results)}")
    print(f"mae {np.mean(mae_fold_results)}")
    results[a] = np.mean(mse_fold_results)

best_alpha = min(results, key=results.get)
print(best_alpha)

























#4
print("--------------------------------------------------")
X_train_final, _ = normalizeData(training_data, training_data)
y_train_final = prices
final_model = Ridge(alpha = best_alpha)
final_model.fit(X_train_final, y_train_final)

print(f"bias: {final_model.intercept_}")
print("coeficienti:", final_model.coef_)

nume_atribute = [
    "anul fabricației", "kilometri", "mileage", "motor", "putere", "numar de locuri", "proprietari", "comb1", "comb2", "comb3", "comb4", "comb5", "manual", "automatic"
]


importanta = pd.DataFrame({
    'Atribut': nume_atribute,
    'Coeficient': final_model.coef_,
    'Importanta_Absoluta': np.abs(final_model.coef_)
})

importanta = importanta.sort_values(by='Importanta_Absoluta', ascending=False)
print("\nTop atribute după influență:")
print(importanta)