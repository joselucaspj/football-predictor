import streamlit as st
import gdown
import os
import joblib
from utils.data_loader import load_match_data
from utils.preprocessor import prepare_data
from utils.model_predictor import predict_all_matches

# Configurações da página
st.set_page_config(layout="wide", page_title="Football Predictor Pro")
# URLs dos modelos (substitua pelos seus IDs reais)
MODEL_URLS = {
    'modelo_predict_gols.pkl': {
        'id': '1XpKUMdD05ZZ70gLDsFaC2wzATm_FCdz7',
        'url': None  # Será construída automaticamente
    },
    'modelo_predict_winner.pkl': {
        'id': '1b_uaLyGSBjxN8oLJMY0-rlXVbMlFu42R',
        'url': None
    }
}
# Construir URLs completas
for model in MODEL_URLS.values():
    model['url'] = f"https://drive.google.com/uc?id={model['id']}&confirm=t"
@st.cache_resource
def load_models():
    """Carrega modelos com tratamento robusto de erros"""
    models = {}
    
    for filename, data in MODEL_URLS.items():
        try:
            if not os.path.exists(filename):
                st.info(f"Baixando {filename}...")
                gdown.download(
                    url=data['url'],
                    output=filename,
                    quiet=False,
                    fuzzy=True
                )
            
            models[filename] = joblib.load(filename)
            st.success(f"{filename} carregado com sucesso!")
            
        except Exception as e:
            st.error(f"Falha ao carregar {filename}: {str(e)}")
            st.stop()  # Interrompe o app se não carregar
    
    return models['modelo_predict_gols.pkl'], models['modelo_predict_winner.pkl']
def main():
    st.title("⚽ Football Predictor Pro")
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("Configurações")
        n_simulations = st.slider("Número de simulações", 1000, 50000, 10000)
        last_n_games = st.slider("Últimos N jogos para análise", 1, 10, 5)
    
    # Carregar dados e modelos
    with st.spinner("Carregando dados e modelos..."):
        df_matches = load_match_data()
        model_gols, model_winner = load_models()
        processed_data = prepare_data(df_matches)
    
    # Processamento principal
    if st.button("🔮 Executar Previsões"):
        with st.spinner("Calculando previsões..."):
            results = predict_all_matches(
                processed_data,
                model_gols,
                model_winner,
                n_simulations=n_simulations,
                last_n_games=last_n_games
            )
        
        # Exibir resultados
        st.subheader("📊 Resultados das Previsões")
        st.dataframe(
            results.style.format({
                'Probability': '{:.2%}',
                'Media_GM_H_HA': '{:.2f}',
                'Media_GS_H_HA': '{:.2f}'
            }),
            height=800,
            use_container_width=True
        )
        
        # Exportar resultados
        st.download_button(
            "💾 Baixar Previsões Completas",
            data=results.to_csv(index=False, encoding='utf-8-sig'),
            file_name="football_predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
