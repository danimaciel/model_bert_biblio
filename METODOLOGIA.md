# Metodologia e referências

Todos os indicadores usam a base após preparação, deduplicação escolhida e filtros.

## Estatísticas e contagens

Média = Σx/n; desvio padrão amostral = √[Σ(x−média)²/(n−1)], indisponível para n<2. Mediana = quantil 0,50; P25, P75 e P90 usam interpolação linear na posição (n−1)p dos valores ordenados. Ausências são excluídas, com N válido e ausentes apresentados separadamente. Produção anual, fontes e tipos contam registros. Cada autor recebe um crédito por documento; palavras-chave contam uma presença por documento. Colaboração = 100 × documentos com >1 autor / documentos com autoria. Percentual temático = 100 × documentos distintos na categoria / documentos filtrados; categorias podem se sobrepor. São definições operacionais do modelo. Referências de implementação: [PANDAS-STD; PANDAS-Q].

## Zipf — frequência de palavras

Forma clássica: f(r)=C/rˢ, com s=1. Nesta implementação, C=T/Σ(1/r), r=1…V, normaliza a curva para T ocorrências e V palavras distintas do corpus observado. Conta todas as ocorrências em título + resumo, inclusive repetições e palavras funcionais. Texto em NFC e casefold; sequências de letras Unicode, sem números, sem lematização, preservando acentos. Hífens/apóstrofos separam palavras. Empates usam ordem alfabética e posições consecutivas. Não se aplica ao ranking de presença de palavras-chave. Textos ausentes não contribuem. A análise abrange somente os campos disponíveis, não o texto integral. [ZIPF; FERRER].

## Lotka — produtividade dos autores

Forma clássica: p(x)=1/[ζ(2)x²]=6/(π²x²), x≥1. Esperado = A×p(x), com A autores distintos e x documentos por autor. Inclui classes intermediárias com zero observado. A tabela vai até o máximo observado; a massa esperada além desse máximo é informada separadamente, sem renormalização. Usa contagem integral de todos os coautores como adaptação operacional; não é reprodução exata do protocolo histórico. Homônimos, variantes e janela temporal afetam resultados. [LOTKA].

## Bradford — dispersão entre fontes

Ordena fontes por produção decrescente (empates por nome). Divide em três zonas contíguas com meta D/3 documentos, sendo D o total com fonte informada. O primeiro corte minimiza a distância acumulada a D/3, reservando duas fontes; o segundo minimiza a distância a 2D/3, reservando uma fonte. Em empate, escolhe o primeiro corte. Não divide fontes; empates de produtividade podem atravessar zonas. Mostra desvio da meta e razões n₂/n₁ e n₃/n₂. Compara n₁:n₂:n₃ com 1:b:b², b=√(n₃/n₁), estimativa descritiva pelas extremidades, não ajuste independente. Exige ao menos três fontes. O campo fonte pode conter livros/eventos: restrinja a periódicos e a um assunto coerente para interpretação clássica. Grafias das fontes não são consolidadas. [BRADFORD].

## Interpretação das leis

Aplicar uma distribuição de referência não demonstra que a base obedece à lei. As curvas são comparações exploratórias com expoentes fixos, sem p-valor, estimação de expoente ou teste de aderência. Bradford exige atenção ao equilíbrio de documentos entre zonas. Amostras pequenas e filtros alteram os padrões; regressão em escala logarítmica isolada não valida uma lei de potência. [CLAUSET].

## Classificação lexical e similaridade

TF-IDF: tf(t,d)×[ln((1+N)/(1+df(t)))+1], seguido de normalização L2. N inclui documentos elegíveis e descrições das categorias no ajuste do vocabulário. Usa unigramas/bigramas, remoção de acentos e lista de palavras comuns definida no modelo. Similaridade cosseno = u·v/(||u||||v||); associa quando score≥limiar escolhido. O score não é probabilidade de acerto. SBERT usa embeddings normalizados do modelo paraphrase-multilingual-MiniLM-L12-v2 e a mesma similaridade. [TFIDF; SBERT; MODEL].

## Descoberta temática

LDA usa matriz de contagens, até 12.000 termos, semente 42 e 20 iterações; o tópico de maior peso determina a associação, excluindo documentos sem vocabulário. BERTopic usa embeddings multilíngues, PCA (até cinco componentes), HDBSCAN e representação c-TF-IDF. PCA substitui a redução usual por UMAP nesta configuração; ruído fica sem categoria e sem score. A descoberta exige ao menos três documentos elegíveis; elegibilidade textual requer 15 caracteres. No modo combinado, descobre apenas no resíduo sem associação. [LDA; BERTOPIC].

## Referências

- **[ZIPF]** Zipf, G. K. (1949). Human Behavior and the Principle of Least Effort. Addison-Wesley. [Acessar fonte](https://archive.org/details/humanbehaviourpr0000zipf)

- **[FERRER]** Ferrer i Cancho, R.; Solé, R. V. (2003). Least effort and the origins of scaling in human language. PNAS, 100(3), 788–791. [Acessar fonte](https://doi.org/10.1073/pnas.0335980100)

- **[LOTKA]** Lotka, A. J. (1926). The frequency distribution of scientific productivity. Journal of the Washington Academy of Sciences, 16(12), 317–323. [Acessar fonte](https://www.jstor.org/stable/24529203)

- **[BRADFORD]** Bradford, S. C. (1934). Sources of information on specific subjects. Engineering, 137, 85–86. Link para republicação em Journal of Information Science, 10(4), 173–180 (1985). [Acessar fonte](https://doi.org/10.1177/016555158501000407)

- **[CLAUSET]** Clauset, A.; Shalizi, C. R.; Newman, M. E. J. (2009). Power-law distributions in empirical data. SIAM Review, 51(4), 661–703. [Acessar fonte](https://doi.org/10.1137/070710111)

- **[PANDAS-STD]** pandas. Series.std: desvio padrão amostral (ddof=1). [Acessar fonte](https://pandas.pydata.org/docs/reference/api/pandas.Series.std.html)

- **[PANDAS-Q]** pandas. Series.quantile: interpolação linear dos quantis. [Acessar fonte](https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html)

- **[TFIDF]** scikit-learn. TfidfTransformer: ponderação TF-IDF e normalização. [Acessar fonte](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

- **[SBERT]** Reimers, N.; Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP. [Acessar fonte](https://arxiv.org/abs/1908.10084)

- **[MODEL]** Sentence Transformers. Ficha do modelo paraphrase-multilingual-MiniLM-L12-v2. [Acessar fonte](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

- **[LDA]** Blei, D. M.; Ng, A. Y.; Jordan, M. I. (2003). Latent Dirichlet Allocation. JMLR, 3, 993–1022. [Acessar fonte](https://www.jmlr.org/papers/v3/blei03a.html)

- **[BERTOPIC]** Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. [Acessar fonte](https://arxiv.org/abs/2203.05794)
