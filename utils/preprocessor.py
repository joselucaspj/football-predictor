import pandas as pd

def standardize_columns(df):
    """Padroniza nomes de colunas"""
    column_map = {
        'Div': 'League',
        'FTHG': 'HomeGoals',
        'FTAG': 'AwayGoals',
        'HG': 'HomeGoals',
        'AG': 'AwayGoals',
        'Home': 'HomeTeam',
        'Away': 'AwayTeam'
    }
    return df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

def prepare_data(data_dict):
    """Prepara todos os dados para análise"""
    # Processar dados históricos
    historical = standardize_columns(data_dict['historical'])
    historical = historical[['League', 'Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']]
    historical['Date'] = pd.to_datetime(historical['Date'], dayfirst=True)
    
    # Processar jogos do dia
    fixtures = standardize_columns(data_dict['fixtures'])
    fixtures = fixtures[['League', 'Date', 'Time', 'HomeTeam', 'AwayTeam']]
    fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
    
    # Adicionar IDs
    fixtures = add_ids(fixtures, data_dict['league_dict'], data_dict['team_dict'])
    historical = add_ids(historical, data_dict['league_dict'], data_dict['team_dict'])
    
    return {
        'historical': historical,
        'fixtures': fixtures
    }

def add_ids(df, league_dict, team_dict):
    """Adiciona IDs de liga e time"""
    df = df.merge(
        league_dict, 
        left_on='League', 
        right_on='League', 
        how='left'
    )
    
    df = df.merge(
        team_dict, 
        left_on='HomeTeam', 
        right_on='Team', 
        how='left'
    ).rename(columns={'Team_ID': 'Home_Team_ID'})
    
    df = df.merge(
        team_dict, 
        left_on='AwayTeam', 
        right_on='Team', 
        how='left'
    ).rename(columns={'Team_ID': 'Away_Team_ID'})
    
    return df
