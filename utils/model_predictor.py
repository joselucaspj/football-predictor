import pandas as pd
import numpy as np
from scipy.stats import poisson
from utils.match_simulator import simulate_match

def predict_all_matches(data_dict, model_gols, model_winner, n_simulations=10000, last_n_games=5):
    """Executa todas as previsões"""
    fixtures = data_dict['fixtures']
    historical = data_dict['historical']
    
    # Calcular médias de gols
    fixtures = calculate_goal_averages(fixtures, historical, last_n_games)
    
    # Prever gols
    features = ['Media_GM_H_HA', 'Media_GS_H_HA', 'Media_GM_A_HA', 'Media_GS_A_HA','Probabilidade_placar_mandante_media_HA',
                'Probabilidade_placar_visitante_media_AH', 'Probabilidade_placar_empate_media_HA',
                '0.0x0.0_Probability_HA','1.0x0.0_Probability_HA','2.0x0.0_Probability_HA','3.0x0.0_Probability_HA',
                '0.0x1.0_Probability_HA','0.0x2.0_Probability_HA','0.0x3.0_Probability_HA','1.0x1.0_Probability_HA',
                '2.0x1.0_Probability_HA','3.0x1.0_Probability_HA','1.0x2.0_Probability_HA','1.0x3.0_Probability_HA',
                '2.0x2.0_Probability_HA','3.0x2.0_Probability_HA','2.0x3.0_Probability_HA','3.0x3.0_Probability_HA',
                'Goleada_Home_Probability_HA','Goleada_Away_Probability_HA','Goleada_Empate_Probability_HA','League_ID','Home_Team_ID','Away_Team_ID'] # Sua lista original de features
    X = fixtures[features]
    fixtures[['Predicted_Goals_H', 'Predicted_Goals_A']] = model_gols.predict(X)
    
    # Simular partidas
    results = []
    num_simulations = 10000
    for _, row in fixtures.iterrows():
         home_goals = row['Predicted_Goals_H']
        away_goals = row['Predicted_Goals_A']
        # Realizando a simulação com o número de simulações ajustado
        simulated_results = simulate_match_predict(home_goals, away_goals, num_simulations=num_simulations)

        # Obtenção dos 8 placares mais prováveis
        top_results = top_results_df(simulated_results, top_n=8)

        for i, result in top_results.iterrows():
            home_goals = int(result['Home_Goals'])
            away_goals = int(result['Away_Goals'])
            probability = result['Probability']
            coluna_placar_H = f'Placar {i}_H'
            coluna_placar_A = f'Placar {i}_A'
            coluna_probabilidade = f'Probabilidade Placar {i}'
            fixtures.at[indice, coluna_placar_H] = home_goals
            fixtures.at[indice, coluna_placar_A] = away_goals
            fixtures.at[indice, coluna_probabilidade] = probability 
    #fixtures.to_csv("teste_aprendizado_de_maquina.csv", index=False)
    
    fixtures['Placar_0_Diff'] = fixtures['Placar 0_H'] - fixtures['Placar 0_A']
    fixtures['Placar_1_Diff'] = fixtures['Placar 1_H'] - fixtures['Placar 1_A']
    fixtures['Placar_2_Diff'] = fixtures['Placar 2_H'] - fixtures['Placar 2_A']
    fixtures['Placar_3_Diff'] = fixtures['Placar 3_H'] - fixtures['Placar 3_A']
    fixtures['Placar_4_Diff'] = fixtures['Placar 4_H'] - fixtures['Placar 4_A']
    fixtures['Placar_5_Diff'] = fixtures['Placar 5_H'] - fixtures['Placar 5_A']
    fixtures['Placar_6_Diff'] = fixtures['Placar 6_H'] - fixtures['Placar 6_A']
    fixtures['Placar_7_Diff'] = fixtures['Placar 7_H'] - fixtures['Placar 7_A']
    
    classification_features = features + ['Placar 0_H', 'Placar 0_A','Probabilidade Placar 0', 'Placar 1_H', 'Placar 1_A','Probabilidade Placar 1', 'Placar 2_H', 'Placar 2_A','Probabilidade Placar 2', 'Placar 3_H', 'Placar 3_A','Probabilidade Placar 3', 'Placar 4_H', 'Probabilidade Placar 4','Placar 5_H', 'Placar 5_A', 'Probabilidade Placar 5', 'Placar 6_H','Placar 6_A', 'Probabilidade Placar 6', 'Placar 7_H', 'Placar 7_A','Probabilidade Placar 7','Placar_0_Diff','Placar_1_Diff','Placar_2_Diff','Placar_3_Diff','Placar_4_Diff','Placar_5_Diff','Placar_6_Diff','Placar_7_Diff']
    pridict_winner = model_winner.predict(fixtures[classification_features])
    fixtures['Predict_winner']= pridict_winner
    fixtures = fixtures[['League', 'Date', 'TIME', 'HomeTeam', 'AwayTeam','Predict_winner']]
    fixtures.columns = ['League', 'Date', 'TIME', 'HomeTeam', 'AwayTeam','Predict_winner']
    
    return fixtures

def calculate_goal_averages(fixtures, historical, n_games):
    """Calcula médias de gols para cada time"""
    
    for col in ['Media_GM_H_HA', 'Media_GS_H_HA', 'Media_GM_A_HA', 'Media_GS_A_HA']:
        fixtures[col] = 0.0
        fixtures['0.0x0.0_Probability_HA'] = 0
        fixtures['1.0x0.0_Probability_HA'] = 0
        fixtures['2.0x0.0_Probability_HA'] = 0
        fixtures['3.0x0.0_Probability_HA'] = 0
        fixtures['0.0x1.0_Probability_HA'] = 0
        fixtures['0.0x2.0_Probability_HA'] = 0
        fixtures['0.0x3.0_Probability_HA'] = 0
        fixtures['1.0x1.0_Probability_HA'] = 0
        fixtures['2.0x1.0_Probability_HA'] = 0
        fixtures['3.0x1.0_Probability_HA'] = 0
        fixtures['1.0x2.0_Probability_HA'] = 0
        fixtures['1.0x3.0_Probability_HA'] = 0
        fixtures['2.0x2.0_Probability_HA'] = 0
        fixtures['3.0x2.0_Probability_HA'] = 0
        fixtures['2.0x3.0_Probability_HA'] = 0
        fixtures['3.0x3.0_Probability_HA'] = 0
        fixtures['Goleada_Home_Probability_HA'] = 0
        fixtures['Goleada_Away_Probability_HA'] = 0
        fixtures['Goleada_Empate_Probability_HA'] = 0
    for idx, row in fixtures.iterrows():
            df_equipe_mandante = historical[((historical['Home_Team_ID'] == fixtures[idx, 'Home_Team_ID']) | (historical['Away_Team_ID'] == fixtures[idx, 'Home_Team_ID'])) & (historical['League_ID'] == fixtures[idx, 'League_ID'])].tail(n)
            df_equipe_visitante = historical[((historical['Home_Team_ID'] == fixtures[idx, 'Away_Team_ID']) | (historical['Away_Team_ID'] == fixtures[idx, 'Away_Team_ID'])) & (historical['League_ID'] == fixtures[idx, 'League_ID'])].tail(n)
            #display(df_equipe)
            if df_equipe.shape[0] == 0:
                fixtures[idx, 'Media_GM_H_HA'] = 0
                fixtures[idx, 'Media_GS_H_HA'] = 0
                fixtures[idx, 'Media_GM_A_HA'] = 0
                fixtures[idx, 'Media_GS_A_HA'] = 0
            else:
                fixtures[idx, 'Media_GM_H_HA'] = round((df_equipe_mandante[(df_equipe_mandante['Home_Team_ID'] == fixtures[idx, 'Home_Team_ID'])]['FTHG'].sum() + df_equipe_mandante[(df_equipe_mandante['Away_Team_ID'] == fixtures[idx, 'Home_Team_ID'])]['FTAG'].sum()) / df_equipe_mandante.shape[0], 2)
                fixtures[idx, 'Media_GS_H_HA'] = round((df_equipe_mandante[(df_equipe_mandante['Home_Team_ID'] == fixtures[idx, 'Home_Team_ID'])]['FTAG'].sum() + df_equipe_mandante[(df_equipe_mandante['Away_Team_ID'] == fixtures[idx, 'Home_Team_ID'])]['FTHG'].sum()) / df_equipe_mandante.shape[0], 2)
                fixtures[idx, 'Media_GM_A_HA'] = round((df_equipe_visitante[(df_equipe_visitante['Home_Team_ID'] == fixtures[idx, 'Away_Team_ID'])]['FTHG'].sum() + df_equipe_visitante[(df_equipe_visitante['Away_Team_ID'] == fixtures[idx, 'Away_Team_ID'])]['FTAG'].sum()) / df_equipe_visitante.shape[0], 2)
                fixtures[idx, 'Media_GS_A_HA'] = round((df_equipe_visitante[(df_equipe_visitante['Home_Team_ID'] == fixtures[idx, 'Away_Team_ID'])]['FTAG'].sum() + df_equipe_visitante[(df_equipe_visitante['Away_Team_ID'] == fixtures[idx, 'Away_Team_ID'])]['FTHG'].sum()) / df_equipe_mandante.shape[0], 2)
            
            simulated_results = simulate_match(row['Media_GM_H_HA'], row['Media_GS_H_HA'], row['Media_GM_A_HA'], row['Media_GS_A_HA'])
            simulated_results = drop_reset_index(simulated_results)
        
            results = top_results_df(simulated_results,100)
            results = drop_reset_index(results)
            resultados = contar_resultados(results)
            fixtures.at[idx,'Probabilidade_placar_mandante_media_HA'] = resultados['Probabilidade placares mandante']
            fixtures.at[idx,'Probabilidade_placar_visitante_media_AH'] = resultados['Probabilidade placares visitante']
            fixtures.at[idx,'Probabilidade_placar_empate_media_HA'] = resultados['Probabilidade placares empate']     
            for i, result in top_results.iterrows():
                home_goals = result['Home_Goals']
                away_goals = result['Away_Goals']
                probability = result['Probability']
        
        
                if home_goals >= 4 and home_goals > away_goals:
                  fixtures.at[index,'Goleada_Home_Probability_HA'] = fixtures.at[index,'Goleada_Home_Probability_HA'] + probability
                elif away_goals >= 4 and home_goals < away_goals:
                  fixtures.at[index,'Goleada_Away_Probability_HA'] = fixtures.at[index,'Goleada_Away_Probability_HA'] + probability
                elif away_goals >= 4 and away_goals >= 4:
                  fixtures.at[index,'Goleada_Empate_Probability_HA'] = fixtures.at[index,'Goleada_Empate_Probability_HA'] + probability
                else:
                  coluna_placar = f'{home_goals}x{away_goals}_Probability_HA'
                  fixtures.at[index, coluna_placar] = probability
        pass
    
    return fixtures


def simulate_match(home_goals_for, home_goals_against, away_goals_for, away_goals_against, num_simulations=10000, random_seed=42):
    np.random.seed(random_seed)
    estimated_home_goals = (home_goals_for + away_goals_against) / 2
    estimated_away_goals = (away_goals_for + home_goals_against) / 2

    home_goals = poisson(estimated_home_goals).rvs(num_simulations)
    away_goals = poisson(estimated_away_goals).rvs(num_simulations)

    results = pd.DataFrame({
        'Home_Goals': home_goals,
        'Away_Goals': away_goals
    })

    return results

def contar_resultados(dataframe):
    vitoria_mandante = 0
    soma_probabilidade_placar_mandante = 0
    vitoria_visitante = 0
    soma_probabilidade_placar_visitante = 0
    empate = 0
    soma_probabilidade_placar_empate = 0

    for indice, linha in dataframe.iterrows():
        if linha['Home_Goals'] > linha['Away_Goals']:
            vitoria_mandante += 1
            soma_probabilidade_placar_mandante = soma_probabilidade_placar_mandante + linha['Probability']
        elif linha['Home_Goals'] < linha['Away_Goals']:
            vitoria_visitante += 1
            soma_probabilidade_placar_visitante = soma_probabilidade_placar_visitante + linha['Probability']
        else:
            empate += 1
            soma_probabilidade_placar_empate = soma_probabilidade_placar_empate + linha['Probability']

    return {'Vitória do Mandante': vitoria_mandante, 'Vitória do Visitante': vitoria_visitante, 'Empate': empate, 'Probabilidade placares mandante': soma_probabilidade_placar_mandante, 'Probabilidade placares visitante': soma_probabilidade_placar_visitante, 'Probabilidade placares empate': soma_probabilidade_placar_empate }

def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df
def top_results_df(simulated_results, top_n):

    result_counts = simulated_results.value_counts().head(top_n).reset_index()
    result_counts.columns = ['Home_Goals', 'Away_Goals', 'Count']

    sum_top_counts = result_counts['Count'].sum()
    result_counts['Probability'] = result_counts['Count'] / sum_top_counts

    return result_counts

def simulate_match_predict(home_goals_for, away_goals_for, num_simulations=10000, random_seed=42):
    np.random.seed(random_seed)

    home_goals = poisson(home_goals_for).rvs(num_simulations)
    away_goals = poisson(away_goals_for).rvs(num_simulations)

    results = pd.DataFrame({
        'Home_Goals': home_goals,
        'Away_Goals': away_goals
    })

    return results
