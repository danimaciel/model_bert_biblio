# Observatório bibliométrico

Aplicação Streamlit em português para análise bibliométrica e categorização temática assistida. Dados de exemplo são fictícios.

## Executar

Python 3.11 ou 3.12 recomendado. No diretório do projeto:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

No Windows, após instalar, também é possível executar `./iniciar.ps1`.

SBERT e BERTopic já estão incluídos no `requirements.txt`. O arquivo abaixo permanece como alternativa equivalente:

```powershell
.venv\Scripts\python -m pip install -r requirements-semantic.txt
```

`requirements-lock.txt` registra o ambiente original Windows/Python 3.12. Para implantação, use `requirements.txt`, que inclui a restrição `pyarrow<25` exigida pelo ambiente de nuvem observado.

A instalação semântica é maior (inclui PyTorch). O primeiro uso baixa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` para `.models`. A análise é local no servidor. A opção separada de nomes conceituais usa uma API de LLM somente por ação explícita. Títulos/resumos longos estão sujeitos ao limite de tokens do encoder; esta versão não faz segmentação de textos completos.

## Fluxo

1. Experimente a demonstração ou envie CSV, TSV ou XLSX (primeira aba, até 30 MB e 20 mil registros).
2. Confira o mapeamento de campos, separador de autores e deduplicação.
3. Explore produção anual, periódicos, autoria, citações e estatísticas.
4. Escolha descoberta, classificação por categorias ou combinação.
5. Revise nomes/definições e rejeite categorias. Nomes iguais unem categorias na visualização; os IDs originais são preservados na exportação.
6. Explore os documentos associados e baixe o ZIP com relatório HTML, gráficos HTML autossuficientes e tabelas CSV.

## Métodos e limites

- **Classificação semântica:** similaridade cosseno entre título + resumo e nome + definição da categoria, com SBERT multilíngue. Todas as associações acima do limiar são mantidas. Similaridade não é probabilidade. Calibre o limiar com uma amostra revisada.
- **Descoberta semântica:** BERTopic com SBERT, PCA (até cinco dimensões), HDBSCAN e representação lexical. Ruído não é forçado a entrar em categoria; pode não haver grupos suficientes. Não substituímos esse resultado silenciosamente por outro método.
- **Modo leve:** LDA com contagens de palavras para descoberta e TF-IDF para classificação lexical. Não é um substituto semântico do SBERT. LDA associa o tópico de maior peso a cada documento; esse peso não é acurácia.
- **Combinado:** classifica primeiro e descobre categorias somente nos documentos não associados. Não descobre temas adicionais em documentos já classificados.
- **Nomes automáticos:** inicialmente baseados nos termos dos grupos. A opção de IA usa OpenAI Responses com saída estruturada para sugerir nome conceitual, definição e justificativa com IDs de artigos. Exige chave e identificador de modelo compatível, informados na interface; pode gerar cobrança. Mostra previamente os trechos enviados (até cinco artigos, resumo limitado a 1.200 caracteres por artigo, até 30 categorias). Chave não é persistida nem exportada; evidências, propostas e modelo são registrados no relatório. Sem chave, a revisão manual continua disponível. Editar a definição após análise não reclassifica; altere a definição de entrada e execute novamente para isso.
- **Autoria:** separação explícita por ponto e vírgula, vírgula ou barra vertical; contagem integral, sem desambiguação. Escolha vírgula somente quando ela separar pessoas; para `Silva, A.; Costa, B.`, escolha ponto e vírgula.
- **Palavras-chave:** separador independente da autoria, com as mesmas opções. Frequência por documento, sem diferenciar maiúsculas e minúsculas e sem contar repetições dentro do mesmo artigo. O ZIP inclui `palavras_chave.csv`.
- **Exemplos:** `examples/base_demonstracao.csv` usa ponto e vírgula nas listas; `examples/base_demonstracao_virgula.csv` usa vírgulas. Ambos estão disponíveis na barra lateral. No CSV com listas por vírgula, essas células ficam entre aspas para preservar as colunas. Confira as listas interpretadas em Documentos e qualidade.
- **Duplicatas:** DOI normalizado; na ausência de DOI, título normalizado + ano válido. Mantém a primeira ocorrência, sem mesclar metadados. Títulos iguais com DOIs distintos são mantidos.
- **Ausências:** valores inválidos/ausentes não viram zero. Anos inteiros entre 1000 e o próximo ano; citações inteiras não negativas. Desvio padrão amostral e quantis do pandas. Sem valores suficientes, estatísticas ficam indisponíveis.
- **Cobertura:** indicadores refletem a base enviada e os filtros, não a produção global dos autores. Não há busca externa de citações, redes de cocitação nem cálculo de índice h global.
- **Revisão:** aceitação se refere à inclusão da categoria; associações continuam sendo sugestões. A revisão individual de associações ainda não está implementada.

Arquivos e resultados permanecem na sessão do Streamlit; apenas o modelo é compartilhado em cache. Não há contas, banco de dados ou persistência de projetos nesta versão. Ao publicar para múltiplos usuários, dimensione recursos e adicione autenticação conforme o ambiente.

## Verificar

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Teste semântico de integração com download inicial do modelo e apenas dados fictícios:

```powershell
.venv\Scripts\python scripts/check_semantic.py
```

O iniciador `iniciar.ps1` abre em http://127.0.0.1:8517, com acesso apenas neste computador. A configuração compartilhada não fixa endereço nem porta, permitindo que o Streamlit Cloud use sua porta padrão 8501.

No Streamlit Community Cloud, escolha o repositório `danimaciel/model_bert_biblio`, branch `codex/versao-inicial`, arquivo `app.py` e Python 3.12. As dependências são instaladas via `requirements.txt`.

Documentação: [Streamlit](https://docs.streamlit.io/), [BERTopic](https://maartengr.github.io/BERTopic/), [Sentence Transformers](https://www.sbert.net/).
