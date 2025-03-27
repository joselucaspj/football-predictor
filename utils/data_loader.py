import pandas as pd
import requests
from io import StringIO

def download_football_data(urls):
    """Baixa dados de múltiplas fontes"""
    all_data = []
    for league_name, league_urls in urls.items():
        for url in league_urls:
            try:
                response = requests.get(url)
                df = pd.read_csv(StringIO(response.text))
                df['League'] = league_name
                all_data.append(df)
            except Exception as e:
                print(f"Erro ao baixar {url}: {str(e)}")
    return pd.concat(all_data) if all_data else pd.DataFrame()

def load_match_data():
    """Carrega todos os dados necessários"""
    # URLs originais (manter seu dicionário)
    urls = {"Inglaterra - Premiere League": [
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",

    ],
    "Escócia - Premiere League": [
        "https://www.football-data.co.uk/mmz4281/2425/SC0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/SC0.csv",
    ],
    "Alemanha - Bundesliga 1": [
        "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
    ],
    "Itália - Serie A": [
        "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    ],
    "Espanha - La Liga": [
        "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    ],
    "França - Primeira divisão": [
        "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
    ],
    "Holanda": [
        "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/N1.csv",
    ],
    "Bélgica": [
        "https://www.football-data.co.uk/mmz4281/2425/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/B1.csv",
    ],
    "Portugal - Liga 1": [
        "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/P1.csv",
    ],
    "Turquia - Liga 1": [
        "https://www.football-data.co.uk/mmz4281/2425/T1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/T1.csv",
    ],
    "Grecia": [
        "https://www.football-data.co.uk/mmz4281/2425/G1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/G1.csv",
    ],
    "Argentina": ["https://www.football-data.co.uk/new/ARG.csv"],
    "Austria": ["https://www.football-data.co.uk/new/AUT.csv"],
    "Brasil": ["https://www.football-data.co.uk/new/BRA.csv"],
    "China": ["https://www.football-data.co.uk/new/CHN.csv"],
    "Dinamarca": ["https://www.football-data.co.uk/new/DNK.csv"],
    "Finlândia": ["https://www.football-data.co.uk/new/FIN.csv"],
    "Irlanda": ["https://www.football-data.co.uk/new/IRL.csv"],
    "Japao": ["https://www.football-data.co.uk/new/JPN.csv"],
    "México": ["https://www.football-data.co.uk/new/MEX.csv"],
    "Noruega": ["https://www.football-data.co.uk/new/NOR.csv"],
    "Polonia": ["https://www.football-data.co.uk/new/POL.csv"],
    "Romenia": ["https://www.football-data.co.uk/new/ROU.csv"],
    "Russia": ["https://www.football-data.co.uk/new/RUS.csv"],
    "Suecia": ["https://www.football-data.co.uk/new/SWE.csv"],
    "Suíça": ["https://www.football-data.co.uk/new/SWZ.csv"],
    "EUA": ["https://www.football-data.co.uk/new/USA.csv"]} 
    
    # Carregar dados históricos
    historical_data = download_football_data(urls)
    
    # Carregar jogos do dia
    fixtures = pd.read_csv("https://www.football-data.co.uk/fixtures.csv")
    
    # Carregar dicionários
    league_dict = pd.read_csv("assets/ligas_dicionario.csv")
    team_dict = pd.read_csv("assets/times_dicionario.csv")
    
    return {
        'historical': historical_data,
        'fixtures': fixtures,
        'league_dict': league_dict,
        'team_dict': team_dict
    }
