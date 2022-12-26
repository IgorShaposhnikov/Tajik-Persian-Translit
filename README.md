# Tajik-Persian-Translit

Таджикско-Персидская Транслитерация основаная на машинном обучении.

Ссылка на готовую модель: https://drive.google.com/drive/folders/1-TKVvtfOhsyWNiAB16q2xxu_acTwcDQP?usp=sharing                       
**Внимание готовая модель имеет очень плохие метрики!!!**

Примеры использования готовой модели:
```py
from simpletransformers.t5 import T5Model

model = T5Model("mt5", "output")
print(model.predict(["аамм"]))
```
