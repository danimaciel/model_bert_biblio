"""Categorias sugeridas, com evidências e atribuição múltipla explícita."""
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

STOP = sorted(set(ENGLISH_STOP_WORDS) | set('a o as os um uma uns umas de da do das dos em no na nos nas por para com sem e ou que se ao aos e sao foi foram ser como mais este esta estes estas esse essa isso sua seu suas seus entre sobre pela pelo pelas pelos estudo estudos resultado resultados objetivo pesquisa artigo presente the'.split()))
MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
ASSIGN_COLUMNS = ['record_id', 'category_id', 'score', 'method']
CAT_COLUMNS = ['category_id', 'name', 'definition', 'origin', 'terms', 'accepted']


def validate_categories(frame):
    frame = frame.copy().fillna('')
    frame['name'] = frame['name'].astype(str).str.strip()
    frame['definition'] = frame['definition'].astype(str).str.strip()
    frame = frame.loc[frame.name.ne('') | frame.definition.ne('')]
    if (frame.name.eq('') | frame.definition.eq('')).any():
        raise ValueError('Preencha nome e definição de cada categoria.')
    if frame.name.str.casefold().duplicated().any():
        raise ValueError('Use nomes diferentes para as categorias.')
    return frame.reset_index(drop=True)


def analyze(df, categories, mode, engine, threshold=.35, n_topics=5, min_size=5, encoder=None):
    valid = df.loc[df.text.str.len().ge(15)].copy()
    if len(valid) < 2:
        raise ValueError('São necessários ao menos dois documentos com título/resumo de 15 caracteres ou mais.')
    categories = validate_categories(categories)
    if mode != 'Descobrir categorias' and categories.empty:
        raise ValueError('Adicione ao menos uma categoria com definição.')
    records, assignments, notes = [], [], []
    texts = valid.text.tolist()
    ids = valid.record_id.tolist()
    semantic = engine == 'SBERT + BERTopic'
    embeddings = None
    if semantic:
        if encoder is None:
            raise ValueError('O modelo SBERT não foi carregado.')
        embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    assigned = np.zeros(len(valid), dtype=bool)
    if mode != 'Descobrir categorias':
        descriptions = (categories.name + '. ' + categories.definition).tolist()
        if semantic:
            cat_emb = encoder.encode(descriptions, normalize_embeddings=True, show_progress_bar=False)
            scores = cosine_similarity(embeddings, cat_emb)
        else:
            vector = TfidfVectorizer(stop_words=STOP, strip_accents='unicode', ngram_range=(1, 2))
            matrix = vector.fit_transform(texts + descriptions)
            scores = cosine_similarity(matrix[:len(texts)], matrix[len(texts):])
            notes.append('Classificação lexical TF-IDF: coincidência de vocabulário; não equivale à classificação semântica SBERT.')
        for j, cat in categories.iterrows():
            cid = f'C{j+1:03d}'
            records.append([cid, cat['name'], cat['definition'], 'Definida pelo usuário', '', True])
            for i in np.flatnonzero(scores[:, j] >= threshold):
                assigned[i] = True
                assignments.append([ids[i], cid, float(scores[i, j]), 'Similaridade semântica' if semantic else 'Similaridade lexical'])
    if mode != 'Classificar por categorias':
        # Modo combinado: descoberta no resíduo ainda não associado.
        indices = np.arange(len(valid)) if mode == 'Descobrir categorias' else np.flatnonzero(~assigned)
        if len(indices) < 3:
            notes.append('Descoberta não executada: menos de três documentos disponíveis para descobrir categorias.')
        else:
            subset = [texts[i] for i in indices]
            vector = CountVectorizer(stop_words=STOP, strip_accents='unicode', max_features=12000)
            if semantic:
                from bertopic import BERTopic
                from sklearn.decomposition import PCA
                from sklearn.cluster import HDBSCAN
                model = BERTopic(embedding_model=encoder, vectorizer_model=vector,
                                 umap_model=PCA(n_components=min(5, len(indices)-1)),
                                 hdbscan_model=HDBSCAN(min_cluster_size=min(min_size, len(indices)), min_samples=1),
                                 calculate_probabilities=False, verbose=False)
                labels, _ = model.fit_transform(subset, embeddings=embeddings[indices])
                terms = {k: [word for word, weight in model.get_topic(k)[:6] if word] for k in set(labels) if k != -1}
                strengths = np.full(len(indices), np.nan)
                notes.append('BERTopic com embeddings multilíngues, PCA e HDBSCAN. Documentos de ruído permanecem sem categoria.')
            else:
                counts = vector.fit_transform(subset)
                k = min(n_topics, len(indices), counts.shape[1])
                model = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=20)
                distribution = model.fit_transform(counts)
                labels = distribution.argmax(axis=1)
                # Não classificar documentos sem vocabulário após remoção de palavras comuns.
                labels[np.asarray(counts.sum(axis=1)).ravel() == 0] = -1
                strengths = distribution.max(axis=1)
                vocab = vector.get_feature_names_out()
                terms = {int(k): vocab[weights.argsort()[-6:][::-1]].tolist() for k, weights in enumerate(model.components_)}
                notes.append('LDA com contagens e semente 42; atribuição ao tópico de maior peso. Peso LDA não é confiança validada.')
            for label in sorted(set(labels)):
                if label == -1:
                    continue
                words = terms[label]
                cid = f'E{int(label)+1:03d}'
                name = ' / '.join(words[:3]).capitalize()
                definition = 'Categoria provisória caracterizada por: ' + ', '.join(words) + '. Revise os artigos antes de interpretar.'
                records.append([cid, name, definition, 'Descoberta na base', ', '.join(words), True])
                for local in np.flatnonzero(np.asarray(labels) == label):
                    assignments.append([ids[indices[local]], cid, float(strengths[local]), 'BERTopic' if semantic else 'Peso LDA'])
    excluded = len(df) - len(valid)
    if excluded:
        notes.append(f'{excluded} documento(s) sem texto suficiente foram excluídos somente da análise temática.')
    return pd.DataFrame(records, columns=CAT_COLUMNS), pd.DataFrame(assignments, columns=ASSIGN_COLUMNS), notes
