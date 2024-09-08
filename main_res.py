import argparse
from pathlib import Path
import pandas as pd
import joblib


def evaluate_model(test_set):
    # Загрузка обученной модели и векторизатора
    model = joblib.load('./models_res/logistic_regression_model.pkl')
    vectorizer = joblib.load('./models_res/tfidf_vectorizer.pkl')

    # Преобразование тестовых данных
    test_set['demands'] = test_set['demands'].fillna('')
    X_test = vectorizer.transform(test_set['demands'])

    # Получение истинных меток
    # y_test = test_set['job_title']

    # Получение предсказаний
    predictions = model.predict(X_test)

    # Расчет F1-меры
    # f1 = f1_score(y_test, predictions, average='weighted')
    # print(f'F1 Score: {f1:.3f}')

    # Добавление предсказанных меток в тестовый набор
    test_set['predicted_job_title'] = predictions

    return test_set


def main(
    test_path: Path,
    output_path: Path,
) -> None:
    df = pd.read_csv(test_path, encoding='utf-8', sep=',', low_memory=False)

    res_df = evaluate_model(df)

    submission = pd.DataFrame(columns=['id', 'job_title', 'task_type'])
    submission['id'] = res_df['id']
    submission['job_title'] = res_df['predicted_job_title']
    submission['task_type'] = 'RES'

    submission.to_csv(output_path, index=False, encoding='utf-8', sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Получение предсказаний профессии')
    parser.add_argument(
        '-i',
        '--input',
        type=Path,
        help='Путь до тестовой выборки (.csv)',
    )
    parser.add_argument(
        '-o',
        '--out',
        type=Path,
        help='Путь до выходного файла (.csv)',
    )
    args = parser.parse_args()

    main(test_path=args.input, output_path=args.out)
