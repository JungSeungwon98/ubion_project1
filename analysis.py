import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def parallel_analysis_scree_plot(data, n_iter=100, random_state=None):
    """
    평행 분석과 스크리 도표를 결합한 시각화 함수
    """
    # 데이터 표준화
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 실제 데이터 요인 분석
    fa = FactorAnalyzer(rotation='varimax', use_smc=True, method = 'principal')
    fa.fit(data_scaled)
    real_eigenvalues, _ = fa.get_eigenvalues()
    
    # 무작위 데이터 생성 및 분석
    n_samples, n_features = data.shape
    random_eigenvalues = np.zeros((n_iter, n_features))

    np.random.seed(random_state)
    for i in range(n_iter):
        random_data = np.random.normal(size=(n_samples, n_features))
        fa.fit(random_data)
        rand_eig, _ = fa.get_eigenvalues()
        random_eigenvalues[i] = rand_eig

    # 평균 및 95th 백분위수 계산
    mean_random_eigen = np.mean(random_eigenvalues, axis=0)
    percentile_95 = np.percentile(random_eigenvalues, 95, axis=0)

    # 시각화 설정
    plt.figure(figsize=(10, 6))
    components = range(1, n_features+1)
    
    # 실제 데이터 스크리 플롯
    plt.plot(components, real_eigenvalues, 'bo-', 
             linewidth=2, label='Actual Data')
    
    # 무작위 데이터 평균/95th
    plt.plot(components, mean_random_eigen, 'r--', 
             linewidth=2, label='Random Mean')
    plt.plot(components, percentile_95, 'g-.', 
             linewidth=2, label='95th Percentile')
    
    # 카이저 기준선
    plt.axhline(y=1, color='k', linestyle='--', label='Kaiser Criterion')
    
    # 교차점 강조
    optimal_factors = np.where(real_eigenvalues > percentile_95)[0]
    if len(optimal_factors) > 0:
        plt.scatter(optimal_factors+1, real_eigenvalues[optimal_factors], 
                    s=200, facecolors='none', edgecolors='m', 
                    label='Optimal Factors')
    
    plt.title('Parallel Analysis Scree Plot')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.xticks(components)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 최적 요인 수 제안
    suggested_factors = len(optimal_factors)
    print(f'Suggested number of factors: {suggested_factors}')
    return suggested_factors


def make_factor_score_df(n,data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    fa = FactorAnalyzer(n_factors=n,method='principal',rotation='varimax')
    fit = fa.fit(scaled_data)
    factor_scores = fa.transform(scaled_data)
    df = pd.DataFrame(factor_scores,columns = [f"factor_{i+1}" for i in range(n)])
    return df