# Info
В этой папке находится перенесенная на Python логика игры [ProductOwnerSim](https://npg-team.itch.io/product-owner-simulator).

Оригинальный код можно скачать по [ссылке](https://drive.google.com/file/d/1xP5APPBfDKOtVx6LSZPrkv7osM1zI8Yx/view?usp=sharing) или посмотреть в приватном [репозитории](https://github.com/denrus99/ProductOwnerSim).

Для запуска исходника игры требуется движок [Godot 3.5.3](https://godotengine.org/download/archive/).

# Лидерборд

Доступ к лидерборду в коде на Godot задается в файле global.gd в функции _ready()

Чтобы создать свой лидерборд, нужно создать профиль на [silentwolf.com](silentwolf.com).
Полученные данные вставить в поля `api_key`, `game_id`, и выбрать `game_version`.

Для фиксации версии игры, она была выложена на [отдельно](https://krutovsky-danya.itch.io/productownersimulator).

Чтобы получить доступ к лидерборду в этой версии игры, свяжитесь с krutovsky.danya@gmail.com

# itch.io

Чтобы выложить свою версию игры на itch.io нужно:
1. Открыть игру в IDE Godot
2. Экспортировать в HTML5
3. Собрать файлы в архив
4. Загрузить архив на itch.io