# Projeto Integrador P15 UNIFAP

## Demanda diária de bicicletas e condições meteorológicas
Autor: Márcio Alberto Monteiro de Brito. Especialização em Projetos em Inteligência Artificial — UNIFAP Digital. Professor da aula P14: Prof. Dr. Adolfo Colares. Ano: 2026.

O projeto integra duas fontes independentes: contagens diárias de aluguel do sistema Capital Bikeshare, disponibilizadas pela UCI, e clima diário ERA5 pela API Open-Meteo. São 731 dias em Washington DC, de 2011 a 2012. A variável alvo é `cnt`, número total de aluguéis por dia. Trata-se de análise retrospectiva condicional ao clima observado. Para previsão antecipada real, é necessário substituir esse clima por previsões disponíveis no momento da decisão e repetir a avaliação.

## Estrutura
- `src/pipeline.py`: coleta, validação, carga PostgreSQL, integração SQL, features, treino e avaliação.
- `sql/schema.sql`: duas tabelas com chaves primárias e restrições e uma view integrada.
- `data/raw/`: fontes originais e manifesto SHA256.
- `data/processed/integrated.csv`: base integrada após engenharia de features.
- `notebooks/Projeto_Integrador_P15.ipynb`: notebook com análise, resultados e conclusões.
- `results/`: métricas, predições, gráficos e resumo verificável.
- `models/best_model.pkl`: pipeline completo do melhor modelo selecionado na validação.
- `docs/`: relatório acadêmico.

## Execução com PostgreSQL
Requer Python 3.11 ou 3.12, Docker com Compose e porta 5432 disponível. Se houver outro PostgreSQL, ajuste a URL em vez de iniciar o serviço Docker.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
# Edite .env: escolha uma senha local e a mesma senha na URL.
docker compose --env-file .env up -d --wait
# Linux/macOS: substitua pela URL preenchida no seu .env
export DATABASE_URL='postgresql+psycopg://projeto:SUA_SENHA@localhost:5432/bicicletas'
# Windows PowerShell: $env:DATABASE_URL='postgresql+psycopg://projeto:SUA_SENHA@localhost:5432/bicicletas'
python src/pipeline.py
python -m jupyterlab
```
Abra `notebooks/Projeto_Integrador_P15.ipynb` e execute todas as células. As fontes originais estão incluídas; `python src/pipeline.py --refresh` renova as coletas. A carga substitui somente os registros das duas tabelas específicas, em uma transação; use banco dedicado ao projeto.

`python src/pipeline.py --offline` permite verificar integração e treinamento sem banco, mas **não satisfaz a etapa PostgreSQL do P15**. O arquivo `results/summary.json` registra se a execução passou pelo banco.

## Método e resultados
Divisão temporal: 584 dias de treino até 06/08/2012 e 147 dias de teste de 07/08/2012 a 31/12/2012. Quatro divisões de validação temporal apenas no treino; escolha por menor RMSE médio. Comparação de Ridge, Random Forest e Gradient Boosting com hiperparâmetros fixos. A média do treino é a referência simples. Transformações são ajustadas dentro dos folds. `casual`, `registered` e o identificador sequencial não entram no modelo. Os dois primeiros somam o alvo e causariam vazamento.

Resultado reproduzido: Gradient Boosting, RMSE 1123,59; MAE 800,52; R² 0,641. Média do treino: RMSE 2560,56. A redução do RMSE é de aproximadamente 56,1%. Consulte o CSV para todos os modelos. O modelo escolhido não foi selecionado pelo teste. Os resíduos e as curvas revelam erros em dias extremos; nenhum resultado garante desempenho futuro.

## Fontes e atribuição
FANAEE-T, Hadi. Bike Sharing. UCI Machine Learning Repository, 2013. DOI: https://doi.org/10.24432/C5W894. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset. Acesso em: 9 out. 2026. Licença CC BY 4.0 (ver página do dataset). Dados diários e calendário extraídos; variáveis meteorológicas originais não utilizadas.

OPEN-METEO. Historical Weather API. https://open-meteo.com/en/docs/historical-weather-api. Acesso em: 9 out. 2026. Weather data by Open-Meteo.com, ERA5. Dados sob CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. Dados diários integrados e transformados; sem endosso dos fornecedores.

## Entrega no Moodle
1. Revise o notebook e o relatório e confirme os nomes de outros integrantes, se houver.
2. Execute o fluxo com PostgreSQL e confirme `postgresql_executed: true` no resumo.
3. Crie um repositório Git e publique os arquivos deste pacote. Não publique `.env`, senhas ou `.venv`.
4. Confira que o professor consegue acessar o repositório (público ou acesso concedido).
5. Cole o link no campo da atividade e anexe `Projeto_Integrador_P15.ipynb`.

O pacote não cria automaticamente um repositório remoto nem envia arquivos ao Moodle. Uma API FastAPI é bônus do P14 e não consta como exigência do P15 nas imagens fornecidas.
