# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import pickle
from datetime import datetime
import logging

# 경고 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="주식 요인 분석 백테스팅",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 적용
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.25rem;
    padding: 1rem;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.25rem;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<h1 class="main-header">📈 주식 요인 분석 기반 백테스팅 시스템</h1>', unsafe_allow_html=True)

# 사이드바 메뉴
st.sidebar.title("🧭 네비게이션")
st.sidebar.markdown("---")
page = st.sidebar.radio("페이지 선택", [
    "🎯 선정된 팩터", 
    "📊 백테스팅", 
    "📋 데이터 업로드",
    "ℹ️ 도움말"
])

# 데이터 로드 함수들
@st.cache_data
def load_factor_data():
    """팩터 분석 관련 데이터 로드"""
    try:
        data_files = {}
        
        # Excel 파일들 로드
        try:
            data_files['factor_loadings'] = pd.read_excel('data/factor_loadings_동행.xlsx', index_col=0)
            logger.info("Factor loadings loaded successfully")
        except FileNotFoundError:
            st.warning("factor_loadings_동행.xlsx 파일을 찾을 수 없습니다.")
            data_files['factor_loadings'] = None
            
        try:
            data_files['factor_score'] = pd.read_excel('data/factor_score_동행.xlsx', index_col=0)
            logger.info("Factor scores loaded successfully")
        except FileNotFoundError:
            st.warning("factor_score.xlsx 파일을 찾을 수 없습니다.")
            data_files['factor_score'] = None
            
        try:
            data_files['stock_returns'] = pd.read_excel('data/kospi200_rtn_동행.xlsx', index_col=0)
            logger.info("Stock returns loaded successfully")
        except FileNotFoundError:
            st.warning("ts_rtn.xlsx 파일을 찾을 수 없습니다.")
            data_files['stock_returns'] = None

        # Pickle 파일 로드
        try:
            with open('data/coef.pickle_동행', 'rb') as f:
                data_files['factor_coefficients'] = pickle.load(f)
                data_files['fa_object'] = pickle.load(f)
            logger.info("Pickle files loaded successfully")
        except FileNotFoundError:
            st.warning("coef.pickle 파일을 찾을 수 없습니다.")
            data_files['factor_coefficients'] = None
            data_files['fa_object'] = None
        except Exception as e:
            st.error(f"Pickle 파일 로드 중 오류: {str(e)}")
            data_files['factor_coefficients'] = None
            data_files['fa_object'] = None

        return data_files
        
    except Exception as e:
        st.error(f"데이터 로드 중 예상치 못한 오류 발생: {str(e)}")
        return {}

def validate_data(data_dict):
    """데이터 유효성 검사"""
    required_files = ['factor_loadings', 'factor_score', 'factor_coefficients', 'stock_returns']
    missing_files = []
    
    for file_key in required_files:
        if data_dict.get(file_key) is None:
            missing_files.append(file_key)
    
    return missing_files

def create_factor_heatmap(factor_loadings):
    """팩터 로딩 히트맵 생성"""
    fig = px.imshow(
        factor_loadings.T,
        title="요인 적재량 히트맵",
        labels=dict(x="재무 변수", y="요인", color="적재량"),
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(height=600, width=1000)
    return fig

def calculate_portfolio_performance(returns_series):
    """포트폴리오 성과 지표 계산"""
    if returns_series.empty or returns_series.isna().all():
        return {}
    
    # 기본 통계
    returns_clean = returns_series.dropna()
    total_return = (1 + returns_clean).prod() - 1
    annualized_return = (1 + total_return) ** (1/len(returns_clean)) - 1
    volatility = returns_clean.std() * np.sqrt(len(returns_clean))
    
    # 샤프 비율 (무위험 수익률 2% 가정)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else np.nan
    
    # 최대 낙폭 (Maximum Drawdown)
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 승률
    win_rate = (returns_clean > 0).mean()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

# 페이지 1: 선정된 팩터
if page == "🎯 선정된 팩터":
    st.header("🎯 선정된 요인과 그 특성")
    st.markdown("""
    재무 데이터에 요인 분석을 적용하여 도출한 핵심 요인들입니다. 
    각 요인이 어떤 재무 변수들과 강하게 연관되어 있는지 확인할 수 있습니다.
    """)
    
    data_dict = load_factor_data()
    factor_loadings = data_dict.get('factor_loadings')
    
    if factor_loadings is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 요인 적재량 (Factor Loadings)")
            
            # 인터랙티브 히트맵
            fig_heatmap = create_factor_heatmap(factor_loadings)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 상위/하위 적재량 표시
            st.subheader("🔝 주요 적재량")
            for factor in factor_loadings.columns:
                with st.expander(f"📈 {factor}"):
                    factor_data = factor_loadings[factor].abs().sort_values(ascending=False)
                    top_5 = factor_data.head(5)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**상위 5개 변수:**")
                        for var, loading in top_5.items():
                            original_loading = factor_loadings.loc[var, factor]
                            color = "🟢" if original_loading > 0 else "🔴"
                            st.write(f"{color} {var}: {original_loading:.3f}")
                    
                    with col_b:
                        # 요인 해석 가이드
                        st.write("**요인 해석 가이드:**")
                        if any(keyword in top_5.index for keyword in ['ROE', 'ROA', 'ROS']):
                            st.write("💰 수익성 관련 요인")
                        elif any(keyword in top_5.index for keyword in ['부채비율', 'Debt']):
                            st.write("🏦 재무안정성 관련 요인")
                        elif any(keyword in top_5.index for keyword in ['매출', 'Sales', 'Revenue']):
                            st.write("📈 성장성 관련 요인")
                        else:
                            st.write("🔍 기타 요인")
        
        with col2:
            st.subheader("📋 데이터 정보")
            st.info(f"**총 요인 수:** {len(factor_loadings.columns)}")
            st.info(f"**재무 변수 수:** {len(factor_loadings.index)}")
            
            # 요인별 설명력
            st.subheader("📊 요인별 영향도")
            factor_importance = factor_loadings.abs().mean().sort_values(ascending=False)
            
            fig_bar = px.bar(
                x=factor_importance.values,
                y=factor_importance.index,
                orientation='h',
                title="요인별 평균 절대 적재량",
                labels={'x': '평균 절대 적재량', 'y': '요인'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # 상관관계 매트릭스
        st.subheader("🔗 요인 간 상관관계")
        correlation_matrix = factor_loadings.T.corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="요인 간 상관관계 매트릭스",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    else:
        st.error("요인 적재량 데이터를 로드할 수 없습니다. 'data' 폴더를 확인해주세요.")

# 페이지 2: 백테스팅
elif page == "📊 백테스팅":
    st.header("📊 주식 요인 기반 백테스팅")
    st.markdown("""
    선정된 요인을 활용하여 포트폴리오를 구성하고 과거 데이터를 기반으로 
    투자 전략의 성과를 시뮬레이션합니다.
    """)
    
    # 데이터 로드 및 검증
    data_dict = load_factor_data()
    missing_files = validate_data(data_dict)
    
    if missing_files:
        st.error(f"다음 파일들이 누락되었습니다: {', '.join(missing_files)}")
        st.markdown("**필요한 파일들:**")
        st.markdown("- data/factor_loadings_동행.xlsx")
        st.markdown("- data/factor_score_동행.xlsx") 
        st.markdown("- data/ts_rtn_동행.xlsx")
        st.markdown("- data/coef.pickle_동행")
        st.stop()
    
    factor_score_df = data_dict['factor_score']
    factor_coefficients = data_dict['factor_coefficients']
    stock_returns_df = data_dict['stock_returns']
    
    # 백테스팅 설정
    st.subheader("⚙️ 백테스팅 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 연도 범위 설정
        min_year = max(factor_score_df['연도'].min(), stock_returns_df['연도'].min())
        max_year = min(factor_score_df['연도'].max(), stock_returns_df['연도'].max())
        
        start_year = st.number_input(
            "시작 연도:", 
            min_value=int(min_year), 
            max_value=int(max_year), 
            value=int(min_year)
        )
        end_year = st.number_input(
            "종료 연도:", 
            min_value=int(min_year), 
            max_value=int(max_year), 
            value=int(max_year)
        )
    
    with col2:
        # 포트폴리오 설정
        top_n_percent = st.slider(
            "상위/하위 선별 비율 (%)", 
            min_value=5, 
            max_value=30, 
            value=10, 
            step=5
        ) / 100
        
        strategy_type = st.selectbox(
            "투자 전략",
            options=["Long-Short", "Long Only", "Short Only"],
            index=0
        )
    
    with col3:
        # 리밸런싱 및 기타 설정
        rebalancing_freq = st.selectbox(
            "리밸런싱 주기",
            options=["연간", "반기", "분기"],
            index=0
        )
        
        transaction_cost = st.number_input(
            "거래비용 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        ) / 100
    
    if start_year >= end_year:
        st.error("시작 연도는 종료 연도보다 작아야 합니다.")
        st.stop()
    
    # 백테스팅 실행
    if st.button("🚀 백테스팅 실행", type="primary"):
        with st.spinner("백테스팅을 실행 중입니다..."):
            try:
                # 데이터 필터링
                filtered_factor_score = factor_score_df[
                    (factor_score_df['연도'] >= start_year) & 
                    (factor_score_df['연도'] <= end_year)
                ].copy()
                
                filtered_stock_returns = stock_returns_df[
                    (stock_returns_df['연도'] >= start_year) & 
                    (stock_returns_df['연도'] <= end_year)
                ].copy()
                
                if filtered_factor_score.empty or filtered_stock_returns.empty:
                    st.error("선택된 기간에 대한 데이터가 없습니다.")
                    st.stop()
                
                # 백테스팅 함수
                def run_backtest(factor_scores, stock_returns, factor_coef, 
                               top_n_percent, strategy_type, transaction_cost):
                    
                    annual_returns = []
                    portfolio_details = []
                    unique_years = sorted(factor_scores['연도'].unique())
                    
                    for year in unique_years:
                        yearly_factors = factor_scores[factor_scores['연도'] == year].set_index('주식코드')
                        yearly_returns = stock_returns[stock_returns['연도'] == year].set_index('주식코드')
                        
                        if yearly_factors.empty or yearly_returns.empty:
                            continue
                        
                        # 요인 컬럼 식별
                        factor_columns = [col for col in yearly_factors.columns if col.startswith('Factor')]
                        
                        # 포트폴리오 점수 계산
                        yearly_factors_clean = yearly_factors[factor_coef.index]
                        portfolio_scores = (yearly_factors_clean * factor_coef).sum(axis=1)
                        
                        # 공통 종목만 선택
                        common_stocks = portfolio_scores.index.intersection(yearly_returns.index)
                        portfolio_scores = portfolio_scores.loc[common_stocks]
                        yearly_returns_clean = yearly_returns.loc[common_stocks]
                        
                        if portfolio_scores.empty:
                            continue
                        
                        # 상위/하위 종목 선정
                        n_stocks = int(len(portfolio_scores) * top_n_percent)
                        
                        long_stocks = portfolio_scores.nlargest(n_stocks).index
                        short_stocks = portfolio_scores.nsmallest(n_stocks).index
                        
                        # 포트폴리오 수익률 계산
                        long_return = yearly_returns_clean.loc[long_stocks, '수익률'].mean() if len(long_stocks) > 0 else 0
                        short_return = yearly_returns_clean.loc[short_stocks, '수익률'].mean() if len(short_stocks) > 0 else 0
                        
                        # 전략별 수익률
                        if strategy_type == "Long-Short":
                            strategy_return = long_return - short_return
                        elif strategy_type == "Long Only":
                            strategy_return = long_return
                        else:  # Short Only
                            strategy_return = -short_return
                        
                        # 거래비용 차감
                        strategy_return -= transaction_cost
                        
                        annual_returns.append({
                            '연도': year,
                            '수익률': strategy_return,
                            '롱포트폴리오수익률': long_return,
                            '숏포트폴리오수익률': short_return
                        })
                        
                        portfolio_details.append({
                            '연도': year,
                            '롱포트폴리오종목수': len(long_stocks),
                            '숏포트폴리오종목수': len(short_stocks),
                            '총종목수': len(portfolio_scores)
                        })
                    
                    return pd.DataFrame(annual_returns).set_index('연도'), pd.DataFrame(portfolio_details).set_index('연도')
                
                # 백테스팅 실행
                returns_df, details_df = run_backtest(
                    filtered_factor_score, 
                    filtered_stock_returns, 
                    factor_coefficients,
                    top_n_percent, 
                    strategy_type, 
                    transaction_cost
                )
                
                if returns_df.empty:
                    st.error("백테스팅 결과를 생성할 수 없습니다.")
                    st.stop()
                
                # 결과 표시
                st.success("✅ 백테스팅 완료!")
                
                # 성과 지표 계산
                performance_metrics = calculate_portfolio_performance(returns_df['수익률'])
                
                # 메트릭 표시
                st.subheader("📈 성과 요약")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "총 수익률", 
                        f"{performance_metrics.get('total_return', 0):.2%}",
                        delta=f"연평균 {performance_metrics.get('annualized_return', 0):.2%}"
                    )
                
                with col2:
                    st.metric(
                        "변동성", 
                        f"{performance_metrics.get('volatility', 0):.2%}"
                    )
                
                with col3:
                    st.metric(
                        "샤프 비율", 
                        f"{performance_metrics.get('sharpe_ratio', 0):.2f}"
                    )
                
                with col4:
                    st.metric(
                        "승률", 
                        f"{performance_metrics.get('win_rate', 0):.1%}"
                    )
                
                # 추가 지표
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "최대 낙폭", 
                        f"{performance_metrics.get('max_drawdown', 0):.2%}"
                    )
                
                with col2:
                    st.metric(
                        "백테스팅 기간", 
                        f"{start_year}-{end_year} ({end_year-start_year+1}년)"
                    )
                
                # 그래프 생성
                st.subheader("📊 성과 차트")
                
                # 누적 수익률 차트
                cumulative_returns = (1 + returns_df['수익률'].fillna(0)).cumprod()
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("누적 수익률", "연간 수익률", "롱/숏 포트폴리오 비교", "드로우다운"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # 누적 수익률
                fig.add_trace(
                    go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, 
                              mode='lines+markers', name='누적 수익률'),
                    row=1, col=1
                )
                
                # 연간 수익률
                fig.add_trace(
                    go.Bar(x=returns_df.index, y=returns_df['수익률'], name='연간 수익률'),
                    row=1, col=2
                )
                
                # 롱/숏 비교
                fig.add_trace(
                    go.Bar(x=returns_df.index, y=returns_df['롱포트폴리오수익률'], name='롱 포트폴리오'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Bar(x=returns_df.index, y=returns_df['숏포트폴리오수익률'], name='숏 포트폴리오'),
                    row=2, col=1
                )
                
                # 드로우다운
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                fig.add_trace(
                    go.Scatter(x=drawdown.index, y=drawdown.values, 
                              mode='lines', name='드로우다운', fill='tonexty'),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, showlegend=True, title_text="포트폴리오 성과 분석")
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 결과 테이블
                st.subheader("📋 상세 결과")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**연도별 수익률**")
                    st.dataframe(returns_df.style.format({
                        '수익률': '{:.2%}',
                        '롱포트폴리오수익률': '{:.2%}',
                        '숏포트폴리오수익률': '{:.2%}'
                    }))
                
                with col2:
                    st.write("**포트폴리오 구성 현황**")
                    st.dataframe(details_df)
                
                # 다운로드 버튼
                st.subheader("💾 결과 다운로드")
                
                # CSV 다운로드
                csv_data = returns_df.to_csv()
                st.download_button(
                    label="📥 수익률 데이터 CSV 다운로드",
                    data=csv_data,
                    file_name=f"backtest_results_{start_year}_{end_year}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"백테스팅 중 오류가 발생했습니다: {str(e)}")
                with st.expander("🔍 상세 오류 정보"):
                    st.exception(e)

# 페이지 3: 데이터 업로드
elif page == "📋 데이터 업로드":
    st.header("📋 데이터 업로드 및 관리")
    st.markdown("""
    필요한 데이터 파일들을 업로드하거나 현재 데이터 상태를 확인할 수 있습니다.
    """)
    
    # 데이터 상태 확인
    st.subheader("📊 현재 데이터 상태")
    
    data_dict = load_factor_data()
    
    status_data = []
    for key, value in data_dict.items():
        if value is not None:
            if isinstance(value, pd.DataFrame):
                status = "✅ 로드됨"
                info = f"Shape: {value.shape}"
            else:
                status = "✅ 로드됨"
                info = f"Type: {type(value).__name__}"
        else:
            status = "❌ 누락"
            info = "파일 없음"
        
        status_data.append({
            "파일명": key,
            "상태": status,
            "정보": info
        })
    
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True)
    
    # 파일 업로드 섹션
    st.subheader("📤 파일 업로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Excel 파일 업로드**")
        
        factor_loadings_file = st.file_uploader(
            "Factor Loadings (Excel)", 
            type=['xlsx', 'xls'],
            key="factor_loadings"
        )
        
        factor_score_file = st.file_uploader(
            "Factor Scores (Excel)", 
            type=['xlsx', 'xls'],
            key="factor_score"
        )
        
        stock_returns_file = st.file_uploader(
            "Stock Returns (Excel)", 
            type=['xlsx', 'xls'],
            key="stock_returns"
        )
    
    with col2:
        st.write("**Pickle 파일 업로드**")
        
        pickle_file = st.file_uploader(
            "Factor Coefficients (Pickle)", 
            type=['pkl', 'pickle'],
            key="pickle_file"
        )
        
        if st.button("📁 샘플 데이터 생성"):
            st.info("샘플 데이터 생성 기능은 개발 중입니다.")
    
    # 데이터 미리보기
    if any([factor_loadings_file, factor_score_file, stock_returns_file]):
        st.subheader("👀 데이터 미리보기")
        
        if factor_loadings_file:
            df = pd.read_excel(factor_loadings_file)
            st.write("**Factor Loadings Preview:**")
            st.dataframe(df.head())
        
        if factor_score_file:
            df = pd.read_excel(factor_score_file)
            st.write("**Factor Scores Preview:**")
            st.dataframe(df.head())
        
        if stock_returns_file:
            df = pd.read_excel(stock_returns_file)
            st.write("**Stock Returns Preview:**")
            st.dataframe(df.head())

# 페이지 4: 도움말
elif page == "ℹ️ 도움말":
    st.header("ℹ️ 사용 가이드")
    
    st.markdown("""
    ## 🎯 주요 기능
    
    ### 1. 선정된 팩터
    - 요인 분석 결과로 도출된 핵심 요인들을 시각화
    - 각 요인의 재무 변수별 적재량을 히트맵으로 표시
    - 요인 간 상관관계 분석
    
    ### 2. 백테스팅
    - 선정된 요인을 기반으로 한 투자 전략 시뮬레이션
    - Long-Short, Long Only, Short Only 전략 지원
    - 다양한 성과 지표 제공 (수익률, 변동성, 샤프 비율, 최대 낙폭 등)
    - 인터랙티브 차트를 통한 결과 시각화
    
    ### 3. 데이터 업로드
    - 필요한 데이터 파일들의 상태 확인
    - 새로운 데이터 파일 업로드 기능
    - 데이터 미리보기 제공
    
    ## 📁 필요한 데이터 파일
    
    프로젝트를 실행하기 위해서는 다음 파일들이 `data/` 폴더에 있어야 합니다:
    
    1. **factor_loadings.xlsx** - 요인 적재량 데이터
    2. **factor_score.xlsx** - 요인 점수 데이터 (연도, 주식코드 포함)
    3. **ts_rtn.xlsx** - 주식 수익률 데이터 (연도, 주식코드, 수익률 포함)
    4. **coef.pickle** - 요인 계수 및 분석 객체
    
    ## 🔧 백테스팅 설정 옵션
    
    ### 기간 설정
    - **시작/종료 연도**: 백테스팅을 수행할 기간
    - 데이터 가용성에 따라 자동으로 범위가 제한됩니다
    
    ### 포트폴리오 설정
    - **상위/하위 선별 비율**: 포트폴리오에 포함할 종목의 비율 (5-30%)
    - **투자 전략**: 
      - Long-Short: 상위 종목 매수, 하위 종목 매도
      - Long Only: 상위 종목만 매수
      - Short Only: 하위 종목만 매도
    
    ### 리스크 관리
    - **거래비용**: 매매 시 발생하는 비용 (0-1%)
    - **리밸런싱 주기**: 포트폴리오 재구성 빈도
    
    ## 📊 성과 지표 설명
    
    - **총 수익률**: 전체 기간 동안의 누적 수익률
    - **연평균 수익률 (CAGR)**: 복리 기준 연평균 수익률
    - **변동성**: 수익률의 표준편차 (리스크 측정)
    - **샤프 비율**: 위험 대비 수익률 (높을수록 좋음)
    - **최대 낙폭**: 최고점 대비 최대 하락폭
    - **승률**: 양의 수익률을 기록한 비율
    
    ## ⚠️ 주의사항
    
    1. **데이터 품질**: 입력 데이터의 품질이 결과에 큰 영향을 미칩니다
    2. **과최적화**: 백테스팅 결과가 미래 성과를 보장하지 않습니다
    3. **거래비용**: 실제 투자 시 추가적인 비용이 발생할 수 있습니다
    4. **시장 환경**: 과거와 다른 시장 환경을 고려해야 합니다
    
    ## 🚨 문제 해결
    
    ### 데이터 로드 오류
    - `data/` 폴더가 올바른 위치에 있는지 확인
    - 파일명이 정확한지 확인 (대소문자 구분)
    - Excel 파일이 손상되지 않았는지 확인
    
    ### 백테스팅 오류
    - 선택된 기간에 충분한 데이터가 있는지 확인
    - 요인 점수와 주식 수익률 데이터의 종목 코드가 일치하는지 확인
    - 컬럼명이 예상된 형식인지 확인 ('연도', '주식코드', '수익률' 등)
    
    ### 성능 이슈
    - 백테스팅 기간을 단축해보세요
    - 브라우저 캐시를 정리해보세요
    - 데이터 크기가 너무 큰 경우 샘플링을 고려해보세요
    
    ## 📞 지원
    
    추가적인 도움이 필요하시면 다음을 참고하세요:
    - GitHub Issues에 문제를 보고해주세요
    - 문서의 FAQ 섹션을 확인해보세요
    - 커뮤니티 포럼에서 질문해보세요
    """)
    
    # 추가: 기술적 세부사항
    with st.expander("🔧 기술적 세부사항"):
        st.markdown("""
        ### 사용된 라이브러리
        - **Streamlit**: 웹 애플리케이션 프레임워크
        - **Pandas**: 데이터 처리 및 분석
        - **NumPy**: 수치 계산
        - **Plotly**: 인터랙티브 시각화
        - **Matplotlib/Seaborn**: 정적 시각화
        
        ### 요인 분석 방법론
        - **Factor Analysis**: 주성분 분석을 통한 차원 축소
        - **RLM (Robust Linear Model)**: 이상치에 강건한 회귀 분석
        - **요인 점수 계산**: 표준화된 요인 점수 생성
        
        ### 백테스팅 방법론
        - **Long-Short Strategy**: 시장 중립적 전략
        - **리밸런싱**: 주기적 포트폴리오 재구성
        - **성과 측정**: 다양한 위험 조정 수익률 지표
        """)
    
    # 추가: 버전 정보
    with st.expander("📋 버전 정보"):
        st.markdown("""
        ### 현재 버전: 2.0.0
        
        **주요 개선사항:**
        - 인터랙티브 차트 추가 (Plotly)
        - 다양한 투자 전략 지원
        - 개선된 UI/UX 디자인
        - 데이터 업로드 기능
        - 상세한 성과 분석
        - 오류 처리 강화
        
        **이전 버전 대비 변경사항:**
        - Matplotlib → Plotly 차트 마이그레이션
        - 성과 지표 추가 (최대 낙폭, 승률 등)
        - 데이터 검증 로직 강화
        - 사용자 경험 개선
        """)

# 사이드바 추가 정보
st.sidebar.markdown("---")
st.sidebar.subheader("📈 프로젝트 정보")
st.sidebar.info("""
**주식 요인 분석 백테스팅**

버전: 2.0.0  
개발자: Factor Analysis Team  
업데이트: 2024년
""")

st.sidebar.subheader("🔗 유용한 링크")
st.sidebar.markdown("""
- [📚 사용법 가이드](#)
- [🐛 버그 신고](#)
- [💡 기능 제안](#)
- [📧 문의하기](#)
""")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>주식 요인 분석 기반 백테스팅 시스템 | 
    Built with ❤️ using Streamlit | 
    © 2024 Factor Analysis Team</p>
</div>
""", unsafe_allow_html=True)