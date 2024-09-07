import re
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import ujson
from transformers import AutoTokenizer, AutoModel
import torch
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')


def pipeline_factory():  # функция, которая вернет пайплайн с предобработкой данных и предсказанием зп
    # разные параметры для предобработки

    column_42 = 'required_drive_license'
    column_75 = 'languageKnowledge'
    column_76 = 'hardSkills'
    column_77 = 'softSkills'
    # колонки, не представляющие значимой ценности (например, все с одинаковым значением или все None)
    # useless_column_id = [0, 12, 14, 15, 17, 49, 50, 57, 58, 59, 60, 72, 73, 74]
    useless_columns = [
        'id', 'code_profession', 'company_code', 'contact_person', 'data_ids',
        'salary_min', 'salary_max', 'vacancy_address_additional_info', 'vacancy_address',
        'vacancy_address_code', 'vacancy_address_house', 'full_company_name', 'company_inn', 'company',
    ]
    # problems = [42, 75, 76, 77]
    df4_drop_cols = [
        'academic_degree', 'accommodation_type', 'additional_premium', 
        'bonus_type', 'measure_type', 'career_perspective', 'change_time', 
        'code_external_system', 'contact_source', 'date_create',
        'date_modify', 'deleted', 'education_speciality', 'foreign_workers_capability', 
        'metro_ids', 'is_mobility_program', 'is_moderated', 'is_uzbekistan_recruitment', 
        'is_quoted', 'oknpo_code', 'okso_code', 'original_source_type', 'publication_period',
        'published_date', 'regionNameTerm', 'required_certificates', 'retraining_capability',
        'retraining_condition', 'retraining_grant', 'retraining_grant_value', 'social_protected_ids',
        'source_type', 'state_region_code', 'status', 'transport_compensation', 'visibility',
        'contactList', 'company_name',
    ]
    # numeric = ['required_experience', 'salary', 'vacancy_address_latitude', 'vacancy_address_longitude', 'work_places']
    # boolean = 'accommodation_capability need_medcard'.split(' ')
    # categorical = 'busy_type code_professional_sphere education regionName company_business_size schedule_type professionalSphereName federalDistrictCode '.split(' ')
    # колонки, используемые для получения эмбеддингов
    text = 'ss hs additional_requirements other_vacancy_benefit position_requirements position_responsibilities vacancy_benefit_ids vacancy_name languages'.split(' ')

    df4__code_professional_sphere__mode = 'Education'
    df4__required_experience__median = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenize_path = Path('./data/nlp/tokenize')  # путь до модели
    model_name = 'ai-forever/sbert_large_mt_nlu_ru'  # название модели
    model_path = Path('./data/nlp/model')

    # Первоначально будет загрузка моделей, потом берутся оффлайн из файла
    if not (tokenize_path / 'vocab.txt').exists():
        AutoTokenizer.from_pretrained(model_name).save_pretrained(str(tokenize_path))

    tokenizer = AutoTokenizer.from_pretrained(str(tokenize_path))

    if not (model_path / 'model.safetensors').exists():
        AutoModel.from_pretrained(model_name).save_pretrained(str(model_path))

    model = AutoModel.from_pretrained(str(model_path)).to(device)

    BSIZE = 256

    # cat_features=[1025, 1026, 1027, 1029, 1030, 1033-1, 1037-1]

    cat_ = CatBoostRegressor()
    cat_.load_model('./catboost_sal/cat_salary_model')  # модель, весь процесс обучения в файле notebook_salary.ipynb

    def language_transform(value: str) -> str:
        res = []
        json = ujson.loads(value)
        for data in json:
            code_language = data['code_language']
            level = data.get('level', 'Любой')
            res.append(f'{code_language} - {level}')
        return ', '.join(res)

    def skills_transform(value: str, key: str) -> str:
        res = []
        json = ujson.loads(value)
        for data in json:
            res.append(data.get(key, ''))
        return ', '.join(res)

    def f1(x: str) -> str:
        x = x.lower()
        x = re.sub(r'(<[^>]*>)', '', x)
        x = re.sub(r'[\.,?!"\';/\-\(\)]|&laquo|&raquo|&nbsp', ' ', x)
        x = re.sub(' +', ' ', x)
        return x

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embedding(sentences: list[str]) -> torch.Tensor:
        encoded_input = tokenizer(
            sentences, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors='pt',
        )
        
        input_ids = encoded_input['input_ids'].to(device)
        token_type_ids = encoded_input['token_type_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        
        with torch.no_grad():
            model_output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'].to(device))
        return sentence_embeddings.cpu()

    def pipeline(df: pd.DataFrame) -> float:
        df1 = df.drop(columns=useless_columns)
        # columns = df.columns

        df1['car_A'] = df1[column_42].str.contains('A')
        df1['car_B'] = df1[column_42].str.contains('B')
        df1['car_C'] = df1[column_42].str.contains('C')
        df1['car_D'] = df1[column_42].str.contains('D')
        df1['car_E'] = df1[column_42].str.contains('E')

        df2 = df1.drop(columns=[column_42])

        df3 = df2#.copy()
        df3['languages'] = df2[column_75].apply(language_transform)

        df4 = df3#.copy()
        df4['hs'] = df3[column_76].apply(lambda s: skills_transform(s, 'hard_skill_name'))
        df4['ss'] = df3[column_77].apply(lambda s: skills_transform(s, 'soft_skill_name'))

        df4.drop(columns=['softSkills', 'hardSkills', 'languageKnowledge'], inplace=True)
        df4.drop_duplicates(inplace=True)
    
        df4.drop(columns=df4_drop_cols, inplace=True)
        
        df4['additional_requirements'] = df4['additional_requirements'].fillna('')
        df4['code_professional_sphere'] = df4['code_professional_sphere'].fillna(df4__code_professional_sphere__mode)
        df4['need_medcard'] = df4['need_medcard'].fillna(False)
        df4['other_vacancy_benefit'] = df4['other_vacancy_benefit'].fillna('')
        df4['position_requirements'] = df4['position_requirements'].fillna('')
        df4['position_responsibilities'] = df4['position_responsibilities'].fillna('')
        df4 = df4.dropna(subset=['regionName'], ignore_index=True)
        df4['required_experience'] = df4['required_experience'].fillna(df4__required_experience__median)
        df4 = df4.dropna(subset=['vacancy_address_latitude'], ignore_index=True)
        df4 = df4.dropna(subset=['vacancy_address_longitude'], ignore_index=True)
        df4['vacancy_benefit_ids'] = df4['vacancy_benefit_ids'].fillna('')
        df4 = df4.dropna(subset=['professionalSphereName'], ignore_index=True)
        df4 = df4.dropna(subset=['federalDistrictCode'], ignore_index=True)
        df4 = df4.drop(columns=['industryBranchName'])
        df4['languages'] = df4['languages'].fillna('')
        df4['hs'] = df4['hs'].fillna('')
        df4['ss'] = df4['ss'].fillna('')

        text_features = df4[text]
        for col in text_features.columns:
            text_features[col] = text_features[col].apply(f1)

        cols = text_features.columns
        text_features['sum'] = np.sum([text_features[col] for col in cols], axis=0)

        embs = []
        data_embs = list(text_features['sum'])

        for i in range(0, len(data_embs), BSIZE):
            embs.append(get_embedding(data_embs[i:i + BSIZE]))

        if len(embs) == 0:
            return None
        concated_embs = torch.concatenate(embs)

        df5 = df4.drop(columns=text)
        df = pd.concat([pd.DataFrame(concated_embs), df5], axis=1)

        df.columns = df.columns.astype(str)

        pred = cat_.predict(df)

        return pred.item()

    return pipeline


def main(
    test_path: Path,
    output_path: Path,
) -> None:
    df = pd.read_csv(test_path, encoding='utf-8', sep=',', low_memory=False)
    if 'salary' in df.columns:
        df = df.drop(columns=['salary'])
    pipeline = pipeline_factory()

    data = []
    
    try:
        for i, row in tqdm(df.iterrows()):
            salary = pipeline(pd.DataFrame([row]))
            if salary is None:
                continue

            data.append({
                'id': row['id'],
                'salary': salary,
                'task_type': 'SAL',
            })
    finally:
        submission = pd.DataFrame(columns=['id', 'salary', 'task_type'], data=data)
        submission.to_csv(output_path, index=False, encoding='utf-8', sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Получение предсказаний заработной платы')
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
