import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

training_file = "Training_data.xlsx"
new_data_file = "Data to AI/Aktau.xlsx"
output_file = "Flood_results/Aktau_r.xlsx"

# Чтение данных
training_data = pd.read_excel(training_file)
new_data = pd.read_excel(new_data_file)

# Разделение на признаки (X) и целевую переменную (y)
X_train = training_data.drop(columns=["Риск паводков"])
y_train = training_data["Риск паводков"]

# Проверка совпадения признаков между обучающими данными и новыми данными (исключая 'Дата')
assert all(col in new_data.columns for col in X_train.columns), \
    "В новых данных отсутствуют некоторые признаки, необходимые для прогноза!"

# Создание и обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Подготовка данных для прогноза
new_data_for_prediction = new_data[X_train.columns]  # Используем только нужные признаки

# Выполнение прогноза
new_data_predictions = model.predict(new_data_for_prediction)

# Добавляем прогнозы в новые данные
new_data["Риск паводков"] = new_data_predictions.round(2)

# Применение сезонных коэффициентов
def apply_seasonal_adjustment(row):
    month = pd.to_datetime(row["Дата"]).month
    if month in [3, 4]:
        return row["Риск паводков"] * 1.8
    elif month in [5]:
        return row["Риск паводков"] * 1.6
    elif month in [12, 1]:
        return row["Риск паводков"] * 0.4
    elif month in [2]:
        return row["Риск паводков"] * 0.5
    else:
        return row["Риск паводков"]

new_data["Риск паводков с коэффициентами"] = new_data.apply(apply_seasonal_adjustment, axis=1).round(2)

new_data = new_data.drop(columns=["Риск паводков"])
new_data.rename(columns={"Риск паводков с коэффициентами": "Риск паводков"}, inplace=True)

# Сохраняем результаты в Excel
new_data.to_excel(output_file, index=False)

print(f"Прогнозы успешно сохранены в файл {output_file}.")
