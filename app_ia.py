import streamlit as st
import pandas as pd
from pypdf import PdfReader
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Codificador Inteligente de Instrumentos", layout="wide")

# --- FUNÇÃO DE MACHINE LEARNING ---
def processar_pdf_com_ml(texto, nome_arquivo, df_treino, limiar):
    frases = re.split(r'[.;]|\n\n', texto)
    
    stopwords_pt = ['o', 'a', 'os', 'as', 'um', 'uma', 'de', 'do', 'da', 'em', 'para', 'com', 'que', 'e']
    vetorizador = TfidfVectorizer(stop_words=stopwords_pt)
    
    # Aprende o vocabulário da sua planilha
    vetores_treino = vetorizador.fit_transform(df_treino['Instrumento'].astype(str))
    registros = []
    
    for frase in frases:
        frase_limpa = frase.strip().replace('\n', ' ')
        if len(frase_limpa) < 20: 
            continue
            
        vetor_frase = vetorizador.transform([frase_limpa])
        similaridades = cosine_similarity(vetor_frase, vetores_treino)[0]
        
        indice_mais_similar = similaridades.argmax()
        maior_similaridade = similaridades[indice_mais_similar]
        
        if maior_similaridade >= limiar:
            registro_similar = df_treino.iloc[indice_mais_similar]
            registros.append({
                "Arquivo": nome_arquivo,
                "Trecho Extraído do PDF": frase_limpa,
                "Instrumento Mais Parecido (Treino)": registro_similar['Instrumento'],
                "Classe": str(registro_similar['Classe']).capitalize(),
                "Tipo": str(registro_similar['Tipo']).capitalize(),
                "Grau de Certeza (%)": round(maior_similaridade * 100, 2)
            })
            
    return registros

# --- INTERFACE VISUAL ---
st.title("🧠 Codificador de Instrumentos com IA")
st.write("Treine o sistema com sua planilha de exemplos e analise PDFs automaticamente.")

with st.sidebar:
    st.header("1. Base de Conhecimento")
    arquivo_csv = st.file_uploader("Upload da Planilha de Treino (CSV)", type="csv")
    st.divider()
    limiar_ia = st.slider("Nível de Exigência da IA", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

if arquivo_csv is not None:
    try:
        df_treino = pd.read_csv(arquivo_csv)
        st.success(f"Base carregada com {len(df_treino)} exemplos!")
    except Exception as e:
        st.error("Erro ao ler o CSV. Certifique-se de que o arquivo está salvo com codificação UTF-8 ou separado por vírgulas.")
        st.stop()

    st.header("2. Processamento de PDFs")
    uploaded_files = st.file_uploader("Suba seus arquivos PDF aqui", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("🚀 Iniciar Análise Inteligente"):
            resultados_gerais = []
            progresso = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    reader = PdfReader(uploaded_file)
                    texto = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                    
                    dados = processar_pdf_com_ml(texto, uploaded_file.name, df_treino, limiar_ia)
                    if dados:
                        resultados_gerais.extend(dados)
                    
                    progresso.progress((i + 1) / len(uploaded_files))
                except Exception as e:
                    st.error(f"Erro ao ler {uploaded_file.name}: {e}")

            if resultados_gerais:
                df_resultados = pd.DataFrame(resultados_gerais)
                st.divider()
                st.success(f"✅ Análise concluída! {len(df_resultados)} trechos classificados.")

                resumo_classe = df_resultados['Classe'].value_counts().reset_index(name='Total')
                resumo_tipo = df_resultados['Tipo'].value_counts().reset_index(name='Total')
                resumo_cruzado = df_resultados.groupby(['Classe', 'Tipo']).size().reset_index(name='Quantidade')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.dataframe(resumo_classe, use_container_width=True, hide_index=True)
                with col2:
                    st.dataframe(resumo_tipo, use_container_width=True, hide_index=True)
                with col3:
                    st.dataframe(resumo_cruzado, use_container_width=True, hide_index=True)

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_resultados.to_excel(writer, sheet_name="Trechos", index=False)
                    resumo_classe.to_excel(writer, sheet_name="Resumo Classe", index=False)
                    resumo_tipo.to_excel(writer, sheet_name="Resumo Tipo", index=False)
                    resumo_cruzado.to_excel(writer, sheet_name="Matriz Cruzada", index=False)
                
                st.download_button(label="📥 Baixar Relatório Excel", data=output.getvalue(), file_name="Relatorio_Codificacao_IA.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                with st.expander("Ver dados detalhados"):
                    st.dataframe(df_resultados)
            else:
                st.warning("Nenhum trecho similar encontrado. Tente diminuir a exigência da IA.")
else:
    st.info("👈 Por favor, faça o upload do seu arquivo CSV de exemplos na barra lateral.")