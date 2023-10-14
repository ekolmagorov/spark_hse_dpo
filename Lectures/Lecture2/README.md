# Полезные ссылки
* [Документация к Flask](https://flask.palletsprojects.com/en/2.2.x/quickstart/)
* [Документация к WTFORMS](https://wtforms.readthedocs.io/en/3.0.x/)
* [Документация к Jinja2](https://jinja.palletsprojects.com/en/3.0.x/templates/)
* [Введение в современный AJAX](https://learn.javascript.ru/fetch)

# Что почитать по теме:
* Flask Web Development: Developing Web Applications with Python by Miguel Grinberg  

# Запуск Flask примера внутри Docker-контейнера
1. Чтобы собрать докер образ:
   ```bash
   docker build -t repo_name/image_name:image_tag .
   ```
2. Чтобы его запустить:
   ```bash
   docker run -p 5000:5000 -v "$PWD/FlaskExample/artifacts:/root/FlaskExample/artifacts" --rm -i repo_name/image_name:image_tag
   ```
