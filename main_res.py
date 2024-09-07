import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm



def main(
    test_path: Path,
    output_path: Path,
) -> None:
    df = pd.read_csv(test_path, encoding='utf-8', sep=',', low_memory=False)

    data = []
    try:
        for i, row in tqdm(df.iterrows()):
            job_title = 'специалист'
            data.append({
                'id': row['id_cv'],
                'job_title': job_title,
                'task_type': 'RES',
            })
    finally:
        submission = pd.DataFrame(columns=['id', 'job_title', 'task_type'], data=data)
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
