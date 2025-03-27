import pandas as pd
import numpy as np
from scipy.stats import poisson

def simulate_match(home_goals_for, home_goals_against, away_goals_for, away_goals_against, n_simulations):
    """Simula uma partida usando Poisson"""
    home = (home_goals_for + away_goals_against) / 2
    away = (away_goals_for + home_goals_against) / 2
    
    home_goals = poisson(home).rvs(n_simulations)
    away_goals = poisson(away).rvs(n_simulations)
    
    return pd.DataFrame({
        'HomeGoals': home_goals,
        'AwayGoals': away_goals
    })

def top_results(simulated, top_n=8):
    """Retorna os placares mais comuns"""
    counts = simulated.value_counts().reset_index()
    counts.columns = ['HomeGoals', 'AwayGoals', 'Count']
    counts['Probability'] = counts['Count'] / counts['Count'].sum()
    return counts.head(top_n)
