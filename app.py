import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from bibliometria.data import FIELDS, read_table, guess_mapping, prepare, descriptive, author_ranking, keyword_ranking
from bibliometria.topics import MODEL, CAT_COLUMNS, ASSIGN_COLUMNS, analyze
from bibliometria.export import export_zip
from bibliometria.labels import evidence_payload, suggest_labels

st.set_page_config(page_title='Observatório bibliométrico', page_icon='📚', layout='wide')
st.markdown('''<style>.block-container{padding-top:4rem;max-width:1400px}h1{letter-spacing:-1.5px}div[data-testid="stMetric"]{background:white;border:1px solid #e0e7ef;padding:18px;border-radius:12px}div[data-testid="stMetricValue"]{color:#087f8c}</style>''', unsafe_allow_html=True)
st.caption('PESQUISA • EVIDÊNCIAS • DESCOBERTA')
st.title('Observatório bibliométrico')
st.write('Da sua base às evidências: explore a produção científica e construa categorias temáticas revisáveis.')


@st.cache_resource
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, cache_folder=str(Path('.models').resolve()))


def chart_counts(series, label, title):
    counts = series.dropna().value_counts().head(15).rename_axis(label).reset_index(name='Documentos')
    fig = px.bar(counts, x='Documentos', y=label, orientation='h', title=title, color_discrete_sequence=['#087F8C'])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


with st.sidebar:
    st.header('Sua base')
    source_kind = st.radio('Origem', ['Demonstração', 'Enviar arquivo'])
    if source_kind == 'Enviar arquivo':
        upload = st.file_uploader('CSV, TSV ou Excel (.xlsx)', type=['csv', 'tsv', 'xlsx'])
        if upload is None:
            st.info('Envie uma exportação com título e, preferencialmente, resumo.')
            st.stop()
        content, filename = upload.getvalue(), upload.name
    else:
        filename = 'base_demonstracao.csv'
        content = Path('examples/base_demonstracao.csv').read_bytes()
        st.caption('Dados fictícios para conhecer a ferramenta.')
    st.download_button('Baixar exemplo de entrada', Path('examples/base_demonstracao.csv').read_bytes(), 'base_exemplo.csv', 'text/csv')
    st.download_button('Baixar exemplo com listas por vírgula', Path('examples/base_demonstracao_virgula.csv').read_bytes(), 'base_exemplo_virgula.csv', 'text/csv')
    st.divider()
    st.caption('A análise funciona localmente no servidor. SBERT baixa um modelo no primeiro uso. A rotulagem opcional por IA envia apenas os trechos apresentados para conferência, quando você a aciona.')

try:
    raw = read_table(content, filename)
except Exception as exc:
    st.error(f'Não foi possível ler a base: {exc}')
    st.stop()
if raw.empty:
    st.warning('O arquivo não contém registros.')
    st.stop()
if len(raw) > 20000:
    st.warning('Esta versão aceita até 20 mil registros por análise. Divida a base antes de enviar.')
    st.stop()

file_id = hashlib.sha256(content).hexdigest()[:12]
with st.expander('1 · Conferir campos e preparar a base', expanded=source_kind == 'Enviar arquivo'):
    st.caption('Mapeamento automático para nomes comuns de Scopus, Web of Science e tabelas próprias. Confira antes de analisar. Excel: primeira aba.')
    guesses = guess_mapping(raw.columns)
    columns = st.columns(3)
    mapping = {}
    options = ['Não disponível'] + list(raw.columns)
    for i, (key, (label, _)) in enumerate(FIELDS.items()):
        choice = columns[i % 3].selectbox(label, options, index=options.index(guesses[key]) if guesses[key] else 0, key=f'map_{file_id}_{key}')
        mapping[key] = None if choice == 'Não disponível' else choice
    separator = st.selectbox('Separador de autores', [';', ',', '|'], help='Use vírgula somente quando ela separar pessoas. Para nomes como Silva, A.; Costa, B., escolha ponto e vírgula.')
    keyword_separator = st.selectbox('Separador de palavras-chave', [';', ',', '|'])
    st.caption('Os separadores das listas são independentes do delimitador das colunas do CSV. Confira a separação na aba Documentos e qualidade.')
    deduplicate = st.checkbox('Remover duplicatas por DOI ou, sem DOI, por título + ano', value=True)
    st.dataframe(raw.head(8), hide_index=True)
chosen = [v for v in mapping.values() if v]
if len(chosen) != len(set(chosen)):
    st.error('Cada coluna deve ser associada a apenas um campo.')
    st.stop()
df, quality = prepare(raw, mapping, separator, deduplicate, keyword_separator)
with st.sidebar:
    years = sorted(df.year.dropna().astype(int).unique().tolist())
    chosen_years = st.multiselect('Filtrar anos (vazio = todos)', years)
    if chosen_years:
        df = df.loc[df.year.isin(chosen_years)].reset_index(drop=True)
    sources = sorted(df.source[df.source.ne('')].unique().tolist())
    chosen_sources = st.multiselect('Filtrar periódicos (vazio = todos)', sources)
    if chosen_sources:
        df = df.loc[df.source.isin(chosen_sources)].reset_index(drop=True)
if df.empty:
    st.warning('Nenhum documento corresponde aos filtros.')
    st.stop()
dataset_key = hashlib.sha256((file_id + json.dumps([mapping, separator, keyword_separator, deduplicate, chosen_years, chosen_sources], sort_keys=True)).encode()).hexdigest()
if st.session_state.get('dataset_key') != dataset_key:
    st.session_state['dataset_key'] = dataset_key
    st.session_state.pop('analysis', None)

metrics = st.columns(4)
metrics[0].metric('Documentos', f'{len(df):,}')
metrics[1].metric('Autores identificados', df.author_list.explode().dropna().nunique())
metrics[2].metric('Periódicos / fontes', df.source.replace('', pd.NA).nunique())
metrics[3].metric('Mediana de citações', f'{df.citations.median():.1f}' if df.citations.notna().any() else 'N/D')
overview, thematic, documents, downloads = st.tabs(['Panorama bibliométrico', 'Categorias temáticas', 'Documentos e qualidade', 'Exportar análise'])
figures = {}
with overview:
    left, right = st.columns(2)
    annual = df.year.dropna().astype(int).value_counts().sort_index().rename_axis('Ano').reset_index(name='Documentos')
    if not annual.empty:
        figures['producao_anual'] = px.bar(annual, x='Ano', y='Documentos', title='Produção ao longo do tempo', color_discrete_sequence=['#087F8C'])
        left.plotly_chart(figures['producao_anual'], width='stretch')
    if df.source.ne('').any():
        figures['periodicos'] = chart_counts(df.source.replace('', pd.NA), 'Periódico', 'Principais periódicos e fontes')
        right.plotly_chart(figures['periodicos'], width='stretch')
    left, right = st.columns(2)
    if df.author_list.map(bool).any():
        figures['autores'] = chart_counts(df.author_list.explode(), 'Autor', 'Autores com mais documentos')
        left.plotly_chart(figures['autores'], width='stretch')
    if df.citations.notna().any():
        figures['citacoes'] = px.histogram(df, x='citations', title='Distribuição de citações', labels={'citations': 'Citações'}, color_discrete_sequence=['#E7A43B'])
        figures['citacoes'].update_yaxes(title='Documentos')
        right.plotly_chart(figures['citacoes'], width='stretch')
    st.subheader('Estatísticas descritivas e de posição')
    if df.keyword_list.map(bool).any():
        figures['palavras_chave'] = chart_counts(df.keyword_list.explode(), 'Palavra-chave', 'Palavras-chave mais frequentes')
        st.plotly_chart(figures['palavras_chave'], width='stretch')
        st.caption('Palavras-chave informadas na base; contagem por documento, sem diferenciar maiúsculas e minúsculas. Não são categorias descobertas pelo modelo.')
    st.dataframe(descriptive(df).round(2), hide_index=True, width='stretch')
    available = df.author_count.dropna()
    if not available.empty:
        st.caption(f'Colaboração: {100 * available.gt(1).mean():.1f}% dos {len(available)} documentos com autoria informada têm mais de um autor.')
    st.caption('Contagem integral: cada autor recebe um documento por publicação. Nomes são mantidos como informados, sem desambiguação de pessoas. Citações refletem a fonte e a data da exportação; ausências não são zeros.')
    if df.type.ne('').any():
        st.dataframe(df.type[df.type.ne('')].value_counts().rename_axis('Tipo').reset_index(name='Documentos'), hide_index=True)

with thematic:
    st.subheader('Construa suas categorias de análise')
    st.write('Descubra temas, investigue categorias definidas por você ou combine as duas abordagens.')
    mode = st.radio('Modo de análise', ['Combinar os dois', 'Descobrir categorias', 'Classificar por categorias'], horizontal=True)
    default = pd.DataFrame([{'name': 'Manejo florestal sustentável', 'definition': 'Práticas de gestão, exploração de baixo impacto, certificação e regeneração de florestas.'}, {'name': 'Conservação da biodiversidade', 'definition': 'Proteção de espécies, habitats, diversidade biológica e restauração de ecossistemas.'}])
    if mode != 'Descobrir categorias':
        category_input = st.data_editor(default, num_rows='dynamic', hide_index=True, width='stretch', column_config={'name': 'Categoria', 'definition': 'Definição: quais assuntos entram?'}, key='category_input')
    else:
        category_input = default.iloc[:0]
    col1, col2 = st.columns(2)
    engine = col1.selectbox('Método', ['LDA + classificação lexical', 'SBERT + BERTopic'])
    if engine.startswith('LDA'):
        st.info('Modo local leve: LDA descobre temas; TF-IDF associa categorias pelo vocabulário. Para correspondência de significado, selecione SBERT + BERTopic.')
    else:
        st.caption('SBERT multilíngue para similaridade; BERTopic para descoberta. Requer requirements-semantic.txt e download inicial do modelo.')
    threshold = col2.slider('Similaridade mínima para categorias definidas', .05, .95, .35 if engine.startswith('SBERT') else .10, .05, key='threshold_' + engine)
    st.caption('Similaridade não é probabilidade de acerto. O limiar precisa ser calibrado com artigos revisados. Um documento pode pertencer a várias categorias definidas.')
    with st.expander('Parâmetros de descoberta'):
        n_topics = st.slider('Número de tópicos LDA', 2, 15, 5)
        min_size = st.slider('Tamanho mínimo de grupo BERTopic', 3, 50, 5)
        st.caption('No modo combinado, a descoberta usa somente documentos ainda sem categoria. Temas descobertos recebem uma associação por documento nesta versão.')
    config = {'mode': mode, 'engine': engine, 'threshold': threshold, 'n_topics': n_topics, 'min_size': min_size, 'categories': category_input.to_dict('records'), 'model': MODEL if engine.startswith('SBERT') else None, 'random_state': 42}
    config_key = json.dumps(config, sort_keys=True, ensure_ascii=False)
    if st.button('Analisar categorias', type='primary'):
        try:
            with st.spinner('Analisando documentos e preparando evidências…'):
                encoder = load_encoder() if engine.startswith('SBERT') else None
                cats, links, notes = analyze(df, category_input, mode, engine, threshold, n_topics, min_size, encoder)
                st.session_state['analysis'] = {'categories': cats, 'assignments': links, 'notes': notes, 'config': config, 'config_key': config_key}
        except ImportError:
            st.error('Dependência não instalada. Execute: .venv\\Scripts\\python -m pip install -r requirements-semantic.txt. A opção LDA está disponível na instalação básica.')
        except Exception as exc:
            st.error(f'Não foi possível concluir a análise: {exc}')
    result = st.session_state.get('analysis')
    if result:
        if result['config_key'] != config_key:
            st.warning('Os controles mudaram. Os resultados abaixo continuam referentes à última execução; clique em Analisar categorias para atualizar.')
        for note in result['notes']:
            st.caption(note)
        with st.expander('Sugerir nomes conceituais com IA (opcional)'):
            st.write('Transforme os temas descobertos em propostas de categorias com nome, definição e justificativa. Esta etapa usa a API OpenAI e tem cobrança na sua conta de API.')
            evidence = evidence_payload(df, result['categories'], result['assignments'])
            st.caption('Serão enviados termos, títulos e até 1.200 caracteres do resumo de cinco artigos por categoria. Confira o conteúdo abaixo. A chave fica apenas na sessão e não entra na exportação.')
            st.json(evidence, expanded=False)
            api_key = st.text_input('Chave da API OpenAI', type='password', key='label_api_key')
            label_model = st.text_input('Identificador do modelo com Responses e Structured Outputs', placeholder='Informe um modelo disponível na sua conta')
            if st.button('Enviar os trechos e sugerir nomes', disabled=not evidence or not api_key or not label_model):
                try:
                    with st.spinner('Gerando propostas de categorias…'):
                        proposals = suggest_labels(evidence, api_key, label_model)
                    for proposal in proposals:
                        mask = result['categories'].category_id.eq(proposal['category_id'])
                        result['categories'].loc[mask, 'name'] = proposal['name']
                        result['categories'].loc[mask, 'definition'] = proposal['definition'] + '\nJustificativa sugerida: ' + proposal['rationale']
                    result['labeling'] = {'provider': 'OpenAI', 'model': label_model, 'evidence': evidence, 'proposals': proposals}
                    result['revision'] = result.get('revision', 0) + 1
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        st.markdown('**Revisar categorias**')
        st.caption('Edite nome e definição; desmarque Aceitar para excluir uma categoria dos resultados. Use o mesmo nome em duas linhas para reuni-las nos gráficos. A revisão de nomes não reclassifica artigos. Nomes descobertos são rótulos baseados em termos, não interpretações validadas por IA.')
        review = st.data_editor(result['categories'], hide_index=True, width='stretch',
            disabled=['category_id', 'origin', 'terms'],
            column_config={'category_id': 'ID', 'name': 'Nome da categoria', 'definition': 'Definição', 'origin': 'Origem', 'terms': 'Termos', 'accepted': 'Aceitar'},
            key='review_' + dataset_key + hashlib.sha256(result['config_key'].encode()).hexdigest()[:8] + str(result.get('revision', 0)))
        active = review.loc[review.accepted & review.name.fillna('').str.strip().ne('')].copy()
        active['name'] = active.name.str.strip()
        links = result['assignments'].loc[result['assignments'].category_id.isin(active.category_id)].copy()
        associated = links.merge(active, on='category_id').merge(df, on='record_id')
        st.session_state['reviewed'] = (review, links)
        a, b, c = st.columns(3)
        a.metric('Categorias aceitas', active.name.nunique())
        b.metric('Documentos associados', links.record_id.nunique())
        c.metric('Sem categoria aceita', len(df) - links.record_id.nunique())
        if not associated.empty:
            distinct = associated.drop_duplicates(['record_id', 'name'])
            figures['categorias'] = chart_counts(distinct.name, 'Categoria', 'Documentos por categoria')
            st.plotly_chart(figures['categorias'], width='stretch')
            st.caption('Um documento pode aparecer em mais de uma categoria; percentuais podem somar mais de 100%.')
            evolution = distinct.dropna(subset=['year']).groupby(['year', 'name']).size().reset_index(name='Documentos')
            if not evolution.empty:
                figures['temas_ano'] = px.line(evolution, x='year', y='Documentos', color='name', markers=True, labels={'year': 'Ano', 'name': 'Categoria'}, title='Evolução das categorias')
                st.plotly_chart(figures['temas_ano'], width='stretch')
            selected = st.selectbox('Explorar evidências da categoria', active.name.unique())
            evidence = associated.loc[associated.name.eq(selected)].drop_duplicates('record_id').sort_values('score', ascending=False, na_position='last')
            for definition in active.loc[active.name.eq(selected), 'definition'].unique():
                st.write(definition)
            st.dataframe(evidence[['record_id', 'title', 'abstract', 'year', 'authors', 'score', 'method']], hide_index=True, width='stretch')
            st.caption('Score: similaridade cosseno para classificação ou peso do tópico LDA. BERTopic não recebe score nesta versão. Ordenação não equivale a validação de relevância.')
            st.dataframe(author_ranking(evidence).head(15), hide_index=True)

with documents:
    st.subheader('Cobertura e qualidade dos dados')
    st.write(f"Entrada: {quality['input_records']} registros · Duplicatas removidas: {quality['removed_duplicates']} · Após filtros: {len(df)}")
    missing = [{'Campo': label, 'Ausentes': int(df[key].isna().sum()) if key in ['year', 'citations'] else int(df[key].eq('').sum())} for key, (label, _) in FIELDS.items()]
    st.dataframe(pd.DataFrame(missing), hide_index=True)
    st.caption(f"Valores numéricos inválidos tratados como ausentes antes dos filtros: {quality['invalid_numeric']}.")
    st.markdown('**Conferir a separação das listas**')
    st.dataframe(df[['title', 'author_list', 'keyword_list']].head(10), column_config={'title': 'Título', 'author_list': 'Autores separados', 'keyword_list': 'Palavras-chave separadas'}, hide_index=True, width='stretch')
    query = st.text_input('Buscar em títulos e resumos')
    shown = df.loc[df.text.str.contains(query, case=False, regex=False)] if query else df
    st.dataframe(shown.drop(columns=['text', 'author_list', 'keyword_list']), hide_index=True, width='stretch')

with downloads:
    st.subheader('Leve os resultados com você')
    st.write('O pacote inclui relatório HTML, gráficos interativos e tabelas CSV, além dos parâmetros da análise. O relatório pode ser aberto no navegador e impresso em PDF.')
    result = st.session_state.get('analysis')
    if result:
        review, links = st.session_state['reviewed']
    else:
        review, links = pd.DataFrame(columns=CAT_COLUMNS), pd.DataFrame(columns=ASSIGN_COLUMNS)
    metadata = {'source_file': filename, 'quality': quality, 'mapping': mapping, 'author_separator': separator, 'keyword_separator': keyword_separator,
                'filters': {'years': chosen_years, 'sources': chosen_sources}, 'deduplicate': deduplicate,
                'thematic': result['config'] if result else None, 'notes': result['notes'] if result else [],
                'labeling': result.get('labeling') if result else None,
                'limitations': ['Categorias descobertas são rótulos provisórios baseados em termos; revisão humana necessária.', 'Autoria por contagem integral sem desambiguação.', 'Citações ausentes não são zeros; contagens dependem da fonte e data de exportação.', 'Gráficos e associações usam apenas categorias aceitas.']}
    package = export_zip(df, review, links, metadata, figures)
    st.download_button('Baixar análise completa (.zip)', package, 'analise_bibliometrica.zip', 'application/zip', type='primary')
    st.download_button('Baixar base tratada (.csv)', df.drop(columns=['text', 'author_list', 'keyword_list']).to_csv(index=False).encode('utf-8-sig'), 'base_tratada.csv', 'text/csv')


