# salary_prediction
Prediction of salary expectations and recommendation of a specialty

Версия python >= 3.11.9.
Необходимо установить зависимости `pip install -r ./req.txt`.

Модели для задач предсказания заработной платы (SAL) и профессии (RES) необходимо скачать по ссылке:
https://drive.google.com/drive/folders/16MGpN3iPvVTNc8WgP9xnE8gQp9xdxlWF?usp=sharing

И разместить в папке `./models_res/`.

Для получения предсказаний написаны скрипты: `main_sal.py` - для задачи SAL, `main_res.py` - для задачи RES.
Скрипт активируется с помощью команд:
`python ./main_sal.py -i path/to/test.csv -o ./submission.csv`
или
`python ./main_res.py -i path/to/test.csv -o ./submission.csv`
