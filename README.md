# Minha API

Este é o MVP da sprint 04 do curso de **Engenharia de Software** da **PUC-Rio**

O objetivo aqui é disponibilizar o código do modelo de aprendizado para aferição da qualidade de vinhos. Modelo também disponível em colab.

Linkendin: https://www.linkedin.com/in/tatianepr/


## Arquitetura do projeto

Foi desenvolvido um frontend em JavaScript que usa as APIs desenvolvidos em Python. 

- Frontend (porta 80) -> https://github.com/Tatianepr/mvp4-frontend
- Componente de APIs (porta 5000) -> https://github.com/Tatianepr/mvp4-backend (esse)
- Modelo desenvolvido no Colab - https://colab.research.google.com/drive/1cYrJVMw-cIkE1tmr9svY5e99cu8regeV?usp=sharing
- Também migrei o modelo para uma aplicação local - https://github.com/Tatianepr/mvp4-modelo

# testes

Foi desenvolvido conjunto de cenários para avaliar a qualidade do modelo.

```
pytest test_modelos.py

```

## Como executar 


Será necessário ter todas as libs python listadas no `requirements.txt` instaladas.
Após clonar o repositório, é necessário ir ao diretório raiz, pelo terminal, para poder executar os comandos descritos abaixo.

> É fortemente indicado o uso de ambientes virtuais do tipo [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html).

PAra criar um ambiente virtual: 

```
python -m virtualenv env
.\venv\Scripts\activate
```

Este comando instala as dependências/bibliotecas, descritas no arquivo `requirements.txt`.
```
pip install -r requirements.txt
```


Para executar a API  basta executar:

executar direto o arquivo .py
