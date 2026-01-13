# ai_insights.py
"""
AI Insights Module for Portfolio Dashboard
Generates educational portfolio analysis using Gemini 2.5 Flash
"""

import os
from dotenv import load_dotenv
from google import genai
import edge_tts
import asyncio
import tempfile
import base64
import streamlit as st
from datetime import datetime, date
import numpy as np
from auth import increment_ai_call


# -------------------------------
# CONFIGURATION
# -------------------------------

load_dotenv()

# Collect all available keys
API_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3")
]
API_KEYS = [k for k in API_KEYS if k]  # Remove empty/missing keys

if not API_KEYS:
    raise ValueError("No GEMINI_API_KEYS found in .env file")

# Model configuration
GEMINI_MODEL = "gemini-2.5-flash-lite"
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 4000,  # Reduced per section
    "stop_sequences": None,
}

# TTS configuration
DEFAULT_VOICE = "en-US-GuyNeural"
DEFAULT_RATE = "+2%"

# -------------------------------
# DATA COLLECTION (ENHANCED)
# -------------------------------

def collect_portfolio_data(
    tickers_list,
    weights,
    start_date,
    end_date,
    metrics,
    returns_data=None,
    alpha=None,
    beta=None,
    benchmark_label=None,
    current_sharpe=None,
    optimized_sharpe=None,
    frontier_position=None,
    optimal_improvement=None,
    target_risk=None,  # ✅ NEW
    monte_carlo_p5=None,  # ✅ NEW
    monte_carlo_p50=None,  # ✅ NEW
    monte_carlo_p95=None,  # ✅ NEW
    monte_carlo_days=None  # ✅ NEW
):
    """
    Collect all portfolio metrics + qualitative context.
    """
    
    # Date handling
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    elif isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())
    else:
        start_dt, end_dt = start_date, end_date
    
    days = (end_dt - start_dt).days
    years = days / 365.25
    
    # Calculate correlation context
    correlation_context = "N/A"
    if returns_data is not None and returns_data.shape[1] >= 2:
        corr_matrix = returns_data.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        if avg_corr > 0.7:
            correlation_context = f"High correlation (avg {avg_corr:.2f}) - assets move together, limited diversification"
        elif avg_corr > 0.4:
            correlation_context = f"Moderate correlation (avg {avg_corr:.2f}) - some diversification benefit"
        else:
            correlation_context = f"Low correlation (avg {avg_corr:.2f}) - good diversification"
    
    # Identify dominant asset
    max_weight_idx = np.argmax(weights)
    dominant_asset = tickers_list[max_weight_idx]
    dominant_weight = weights[max_weight_idx] * 100
    
    # Build data dictionary
    data = {
        # Portfolio basics
        "tickers": ", ".join(tickers_list),
        "num_assets": len(tickers_list),
        "weights_display": ", ".join([f"{t}: {w*100:.1f}%" for t, w in zip(tickers_list, weights)]),
        "dominant_asset": f"{dominant_asset} ({dominant_weight:.1f}%)",
        "date_range": f"{start_dt.strftime('%b %Y')} to {end_dt.strftime('%b %Y')}",
        "time_period_years": f"{years:.1f}",
        
        # Risk metrics
        "volatility": f"{metrics.get('volatility_pct', 0):.2f}",
        "sharpe": f"{metrics.get('sharpe', 0):.2f}",
        "sortino": f"{metrics.get('sortino', 0):.2f}" if 'sortino' in metrics else "N/A",
        "max_drawdown": f"{metrics.get('max_drawdown_pct', 0):.2f}",
        "var_95": f"{metrics.get('var_95_pct', 0):.2f}",
        "cvar_95": f"{metrics.get('cvar_95_pct', 0):.2f}" if 'cvar_95_pct' in metrics else "N/A",
        
        # Contextual insights
        "correlation_context": correlation_context,
        
        # Benchmark
        "has_benchmark": alpha is not None,
        "benchmark_name": benchmark_label if benchmark_label else "N/A",
        "alpha": f"{alpha:.2f}" if alpha is not None else "N/A",
        "beta": f"{beta:.2f}" if beta is not None else "N/A",
        
        # Optimization
        "has_optimization": current_sharpe is not None and optimized_sharpe is not None,
        "current_sharpe_opt": f"{current_sharpe:.2f}" if current_sharpe else "N/A",
        "optimized_sharpe_opt": f"{optimized_sharpe:.2f}" if optimized_sharpe else "N/A",
        "sharpe_improvement": f"{(optimized_sharpe - current_sharpe):.2f}" if (current_sharpe and optimized_sharpe) else "N/A",
        
        # Frontier
        "has_frontier": frontier_position is not None,
        "frontier_position": frontier_position if frontier_position else "N/A",
        "optimal_improvement": f"{optimal_improvement:.2f}" if optimal_improvement else "N/A",
        "target_risk": f"{target_risk:.1f}" if target_risk else "N/A",  # ✅ NEW
        
        # ✅ NEW: Monte Carlo
        "has_monte_carlo": monte_carlo_p5 is not None,
        "monte_carlo_days": f"{monte_carlo_days}" if monte_carlo_days else "N/A",
        "monte_carlo_p5": f"{monte_carlo_p5:.2f}" if monte_carlo_p5 else "N/A",
        "monte_carlo_p50": f"{monte_carlo_p50:.2f}" if monte_carlo_p50 else "N/A",
        "monte_carlo_p95": f"{monte_carlo_p95:.2f}" if monte_carlo_p95 else "N/A",
    }
    
    return data
# -------------------------------
# MODULAR PROMPT SYSTEM
# -------------------------------

def create_quick_summary_prompt(data):
    """
    Improved sidebar summary prompt (150-180 words).
    """
    
    prompt = f"""
You are a portfolio analyst providing an executive briefing to a client.

PORTFOLIO:
{data['weights_display']}
Period: {data['date_range']} ({data['time_period_years']} years)

KEY METRICS:
- Volatility: {data['volatility']}% (annualized)
- Sharpe: {data['sharpe']} | Sortino: {data['sortino']}
- Max Drawdown: {data['max_drawdown']}%
- Correlation: {data['correlation_context']}

TASK: Write a 170-word summary addressing:

1. **Portfolio Character** (2 sentences): What type of portfolio is this? What's its strategic DNA?

2. **Risk Profile** (2 sentences): How risky is this portfolio? What's the standout risk metric?

3. **Performance Quality** (1 sentence): Is the risk-adjusted return strong (Sharpe/Sortino interpretation)?

4. **Key Opportunity** (1-2 sentences): What's the #1 actionable insight for improvement?

TONE: Direct, analytical, conversational. Like a portfolio manager briefing a client.
NO FLUFF: Every sentence must deliver value.
NO ADVICE: Frame as "this suggests..." not "you should..."
No greetings or sign-offs. Just a one liner summary.
TARGET: Exactly 170 words. Count carefully.
"""
    
    return prompt


def create_section_1_prompt(data):
    """Section 1: Portfolio Architecture & Risk Analytics (180-200 words)"""
    
    prompt = f"""
You are writing SECTION 1 of a comprehensive portfolio analysis report.

PORTFOLIO DATA:
- Holdings: {data['weights_display']}
- Dominant Position: {data['dominant_asset']}
- Period: {data['date_range']} ({data['time_period_years']} years)
- Volatility: {data['volatility']}% | Sharpe: {data['sharpe']} | Sortino: {data['sortino']}
- Max Drawdown: {data['max_drawdown']}% | VaR(95%): {data['var_95']}%
- Correlation: {data['correlation_context']}

TASK: Write EXACTLY 180-200 words covering:

**Portfolio Composition (90-100 words)**
- Portfolio classification: growth/value/sector-focused?
- Concentration risk from {data['dominant_asset']}
- Correlation dynamics: {data['correlation_context']}
- Systemic risk exposure

**Core Risk Metrics (90-100 words)**
- Volatility ({data['volatility']}%): high/moderate/low? Calculate 1-SD monthly range.
- Sharpe ({data['sharpe']}) vs Sortino ({data['sortino']}): difference meaning
- Drawdown ({data['max_drawdown']}%): recovery % needed

RULES:
- Start with "This portfolio demonstrates..." (NO greetings)
- NO section headers/numbers in output
- MUST be 180-200 words total (COUNT CAREFULLY)
- Complete BOTH parts fully
- Stop writing after 200 words MAXIMUM
"""
    
    return prompt


def create_section_2_prompt(data):
    """Section 2: Tail Risk Analysis (140-160 words)"""
    
    prompt = f"""
You are continuing a portfolio analysis report. Write SECTION 2.

RISK DATA:
- VaR(95%): {data['var_95']}%
- CVaR(95%): {data['cvar_95']}%
- Max Drawdown: {data['max_drawdown']}%
- Volatility: {data['volatility']}%

TASK: Write EXACTLY 140-160 words on tail risk:

**VaR & CVaR Analysis**
- VaR ({data['var_95']}%): practical meaning
- CVaR ({data['cvar_95']}%): why more important than VaR
- Gap between VaR/CVaR: tail risk implications
- Real-world: how many bad days per year (252 trading days)?
- Risk acceptability assessment

RULES:
- Reference "Building on the volatility analysis" to connect to Section 1
- NO greetings, NO section headers
- MUST be 140-160 words (COUNT CAREFULLY)
- Complete full analysis
- Stop writing after 160 words MAXIMUM
"""
    
    return prompt

def create_section_3_prompt(data):
    """Section 3: Benchmark Comparison & Monte Carlo (200 words HARD LIMIT)"""
    
    # Build the prompt dynamically based on what data exists
    benchmark_section = ""
    if data['has_benchmark']:
        benchmark_section = f"""
**Benchmark Analysis (100 words)**
- Benchmark: {data['benchmark_name']}
- Alpha: {data['alpha']}%
- Beta: {data['beta']}

Interpret alpha (skill/luck/risk?), calculate expected loss in -15% market crash using beta, explain systematic vs idiosyncratic risk split, assess if alpha justified by risk taken.
"""
    
    monte_carlo_section = ""
    if data['has_monte_carlo']:
        monte_carlo_section = f"""
**Monte Carlo Simulation (100 words)**
- Simulation period: {data['monte_carlo_days']} trading days
- 5th percentile: {data['monte_carlo_p5']}x
- Median: {data['monte_carlo_p50']}x
- 95th percentile: {data['monte_carlo_p95']}x

Explain what Monte Carlo simulation reveals about future uncertainty. Interpret the range between worst-case ({data['monte_carlo_p5']}x) and best-case ({data['monte_carlo_p95']}x). What does the median outcome ({data['monte_carlo_p50']}x) suggest about expected portfolio growth? Discuss confidence intervals and investment planning implications.
"""
    
    # If neither exists, skip this section entirely
    if not data['has_benchmark'] and not data['has_monte_carlo']:
        return None
    
    prompt = f"""
You are writing SECTION 3 of a portfolio analysis report.

DATA:
{benchmark_section}
{monte_carlo_section}

CRITICAL RULES:
- Write 200-220 words (no more, no less)
- Count every single word as you write
- STOP IMMEDIATELY at word 220 with a complete sentence
- If approaching 220 words mid-sentence, finish that sentence early

TASK (200 words total):
Cover ALL sections provided above. Allocate roughly 100 words per section if both exist, or 200 words if only one exists.

Start with: "Having established the portfolio's intrinsic risk characteristics..."
NO greetings. NO headers. Just analysis.

BEGIN YOUR 200-WORD RESPONSE NOW:
"""
    
    return prompt

def create_section_4_prompt(data):
    """Section 4: Optimization Analysis (150-170 words) - CONDITIONAL"""
    
    prompt = f"""
You are continuing a portfolio analysis report. Write SECTION 4.

OPTIMIZATION DATA:
- Current Sharpe: {data['current_sharpe_opt']}
- Optimized Sharpe: {data['optimized_sharpe_opt']}
- Improvement: {data['sharpe_improvement']}
- Correlation: {data['correlation_context']}

TASK: Write EXACTLY 150-170 words on optimization:

**Efficiency Gains**
- Why {data['sharpe_improvement']} Sharpe improvement occurred
- How optimizer used correlation patterns
- Modern Portfolio Theory: efficient set concept
- Weight changes logic (reduce correlated, increase diversifiers)
- Practical return implications over time

RULES:
- Reference "Building on the risk assessment" to connect
- NO greetings, NO section headers
- MUST be 150-170 words (COUNT CAREFULLY)
- Complete all points
- Stop writing after 170 words MAXIMUM
"""
    
    return prompt


def create_section_5_prompt(data):
    """Section 5: Efficient Frontier Position (130-150 words) - CONDITIONAL"""
    
    prompt = f"""
You are continuing a portfolio analysis report. Write SECTION 5.

FRONTIER DATA:
- Position: {data['frontier_position']} the efficient frontier
- Improvement potential: {data['optimal_improvement']}%
- Current Sharpe: {data['sharpe']}

TASK: Write EXACTLY 130-150 words on frontier position:

**Frontier Analysis**
- Efficient frontier definition (Markowitz in plain language)
- Being "{data['frontier_position']}" frontier: economic meaning
- Return gap: {data['optimal_improvement']}% left on table
- Why below-frontier portfolios are suboptimal
- Adjustment path to efficiency

RULES:
- Reference "The optimization analysis revealed" to connect
- NO greetings, NO section headers
- MUST be 130-150 words (COUNT CAREFULLY)
- Complete all points
- Stop writing after 150 words MAXIMUM
"""
    
    return prompt


def create_section_conclusion_prompt(data, has_benchmark, has_optimization, has_frontier):
    """Final Section: Strategic Summary (120-140 words)"""
    
    context_parts = ["risk metrics"]
    if has_benchmark:
        context_parts.append(f"benchmark comparison against {data['benchmark_name']}")
    if has_optimization:
        context_parts.append("optimization analysis")
    if has_frontier:
        context_parts.append("efficient frontier positioning")
    
    context_string = ", ".join(context_parts)
    
    prompt = f"""
You are writing the FINAL SECTION of a portfolio analysis report.

PREVIOUS SECTIONS: {context_string}

PORTFOLIO SUMMARY:
- Holdings: {data['weights_display']}
- Sharpe: {data['sharpe']} | Sortino: {data['sortino']}
- Volatility: {data['volatility']}% | Drawdown: {data['max_drawdown']}%
{"- Alpha: " + data['alpha'] + "% vs " + data['benchmark_name'] if has_benchmark else ""}
{"- Optimization: +" + data['sharpe_improvement'] + " Sharpe" if has_optimization else ""}
{"- Frontier: " + data['frontier_position'] + " (+" + data['optimal_improvement'] + "% potential)" if has_frontier else ""}

TASK: Write EXACTLY 120-140 words as CONCLUSION:

**Strategic Summary**
- Overall risk-return verdict
- **3 Key Strengths** (with evidence)
- **3 Primary Vulnerabilities** (with evidence)
- Final synthesis statement

RULES:
- Start with "In synthesis" to signal conclusion
- NO greetings, NO section headers
- MUST be 120-140 words (COUNT CAREFULLY)
- This is the FINAL section
- Stop writing after 140 words MAXIMUM
"""
    
    return prompt

# -------------------------------
# API CALL WITH KEY ROTATION
# -------------------------------

def call_gemini_with_rotation(prompt, section_name=""):
    """Internal helper to rotate keys if quota is hit."""
    last_error = None
    
    for i, key in enumerate(API_KEYS):
        try:
            client = genai.Client(api_key=key, http_options={'api_version': 'v1'})
            
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=GENERATION_CONFIG
            )
            return response.text
        
        except Exception as e:
            last_error = e
            if "429" in str(e) or "ResourceExhausted" in str(e) or "quota" in str(e).lower():
                if section_name:
                    st.sidebar.warning(f"Rate limit hit on {section_name}, rotating key...")
                continue
            else:
                raise e
                
    raise Exception(f"All {len(API_KEYS)} API keys exhausted. Last error: {last_error}")


# -------------------------------
# CACHED GENERATION FUNCTIONS
# -------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def generate_quick_summary(data_dict):
    try:
        prompt = create_quick_summary_prompt(data_dict)
        text = call_gemini_with_rotation(prompt)

        # ✅ increment sheet counter
        try:
            email = st.session_state.get("user_email")
            users_ws = st.session_state.get("users_ws")
            if users_ws and email:
                increment_ai_call(users_ws, email, call_type="quick")
        except Exception:
            pass

        return text

    except Exception as e:
        return f"Error generating summary: {str(e)}"


@st.cache_data(ttl=86400, show_spinner=False)  # ✅ 24 hours cache
# Add user email to the input
def generate_full_analysis(data_dict, user_email=None):
    # Cache will be unique per user automatically    
    """
    Generate full analysis by calling API multiple times for each section,
    then combining the results into a cohesive report.
    """
    try:
        sections = []

        # ✅ Count how many calls are made inside this full analysis run
        full_call_count = 0

        def call_and_count(prompt, section_name):
            nonlocal full_call_count
            full_call_count += 1
            return call_gemini_with_rotation(prompt)

        # Section 1
        section_1 = call_and_count(create_section_1_prompt(data_dict), "Section 1")
        sections.append(section_1.strip())

        # Section 2
        section_2 = call_and_count(create_section_2_prompt(data_dict), "Section 2")
        sections.append(section_2.strip())

        # Section 3 (conditional)
        section_3_prompt = create_section_3_prompt(data_dict)
        if section_3_prompt:
            section_3 = call_and_count(section_3_prompt, "Section 3")
            sections.append(section_3.strip())

        # Section 4 (conditional)
        if data_dict.get("has_optimization"):
            section_4 = call_and_count(create_section_4_prompt(data_dict), "Section 4")
            sections.append(section_4.strip())

        # Section 5 (conditional)
        if data_dict.get("has_frontier"):
            section_5 = call_and_count(create_section_5_prompt(data_dict), "Section 5")
            sections.append(section_5.strip())

        # Conclusion
        conclusion = call_and_count(
            create_section_conclusion_prompt(
                data_dict,
                data_dict.get("has_benchmark") or data_dict.get("has_monte_carlo"),
                data_dict.get("has_optimization"),
                data_dict.get("has_frontier"),
            ),
            "Conclusion"
        )
        sections.append(conclusion.strip())

        # -------------------------
        # ✅ Format nicely
        # -------------------------
        section_number = 1
        formatted_sections = []

        formatted_sections.append(f"## {section_number}. Portfolio Architecture & Risk Analytics\n\n{sections[0]}")
        section_number += 1

        formatted_sections.append(f"## {section_number}. Tail Risk & Extreme Scenarios\n\n{sections[1]}")
        section_number += 1

        current_idx = 2

        if section_3_prompt:
            section_title = "Analysis"
            if data_dict.get("has_benchmark") and data_dict.get("has_monte_carlo"):
                section_title = "Benchmark Comparison & Monte Carlo Simulation"
            elif data_dict.get("has_benchmark"):
                section_title = "Benchmark Comparison & Market Sensitivity"
            elif data_dict.get("has_monte_carlo"):
                section_title = "Monte Carlo Simulation & Future Scenarios"

            formatted_sections.append(f"## {section_number}. {section_title}\n\n{sections[current_idx]}")
            section_number += 1
            current_idx += 1

        if data_dict.get("has_optimization"):
            formatted_sections.append(f"## {section_number}. Optimization Analysis\n\n{sections[current_idx]}")
            section_number += 1
            current_idx += 1

        if data_dict.get("has_frontier"):
            formatted_sections.append(f"## {section_number}. Efficient Frontier Position\n\n{sections[current_idx]}")
            section_number += 1
            current_idx += 1

        formatted_sections.append(f"## {section_number}. Strategic Summary\n\n{sections[current_idx]}")

        final_report = "\n\n---\n\n".join(formatted_sections)

        disclaimer = """
**Disclaimer:** *This analysis is generated by AI for educational purposes only. 
Financial markets involve significant risk. Past performance is not indicative of future results. 
Please consult a certified financial advisor before making any investment decisions.*
        """.strip()

        # ✅ Increment Google Sheet counters ONCE per full analysis request
        # (but with the correct number of internal calls)
        try:
            email = st.session_state.get("user_email")
            users_ws = st.session_state.get("users_ws")
            if users_ws and email:
                # increment "full" multiple times since full analysis = multiple calls
                for _ in range(full_call_count):
                    increment_ai_call(users_ws, email, call_type="full")
        except Exception:
            pass

        return f"{final_report}\n\n---\n\n{disclaimer}"

    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# -------------------------------
# TTS & UI
# -------------------------------

@st.cache_data(show_spinner=False)
def generate_tts_audio(text, voice=DEFAULT_VOICE, rate=DEFAULT_RATE):
    # ✅ Clean markdown symbols before TTS
    text_cleaned = text.replace("###", "").replace("##", "").replace("#", "")
    text_cleaned = text_cleaned.replace("**", "").replace("*", "")
    text_cleaned = text_cleaned.replace("---", "")  # remove dividers
    text_cleaned = text_cleaned.replace("```", "")  # remove code blocks
    text_cleaned = text_cleaned.replace("`", "")
    
    # Remove multiple spaces/newlines
    text_cleaned = " ".join(text_cleaned.split())
    
    async def _generate():
        communicate = edge_tts.Communicate(text_cleaned, voice, rate=rate)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            await communicate.save(tmpfile.name)
            audio_bytes = open(tmpfile.name, "rb").read()
            return base64.b64encode(audio_bytes).decode()
    
    try:
        return asyncio.run(_generate())
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def render_sidebar_summary(data_dict):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Quick Summary")
    
    if "ai_quick_summary" not in st.session_state or st.session_state["ai_quick_summary"] is None:
        if st.sidebar.button("💡 Generate Summary", key="generate_summary"):
            with st.sidebar:
                with st.spinner("Generating AI summary..."):
                    summary = generate_quick_summary(data_dict)
                    st.session_state["ai_quick_summary"] = summary
                    st.rerun()
    else:
        st.sidebar.info(st.session_state["ai_quick_summary"])
    
    with st.sidebar:
        if st.button("🔄 Refresh Summary", key="refresh_summary"):
            generate_quick_summary.clear()
            with st.spinner("Regenerating summary..."):
                summary = generate_quick_summary(data_dict)
                st.session_state["ai_quick_summary"] = summary
                st.rerun()
        
        if st.button("📊 Read Aloud", key="sidebar_read_aloud"):
            with st.spinner("Generating audio..."):
                b64_audio = generate_tts_audio(st.session_state["ai_quick_summary"])
                if b64_audio:
                    audio_bytes = base64.b64decode(b64_audio)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)


def render_full_analysis_tab(data_dict):
    st.header("🤖 Comprehensive AI Portfolio Analysis")
    st.write("Please Review the short summary on the sidebar for a quick overview.Also, check all the sections first and post finalization of all sections, generate the full analysis.")
    st.write("""
    Get a detailed, modular institutional-grade analysis covering portfolio composition, 
    risk mechanics, performance attribution, and strategic positioning.
    """)
    
    # Show what sections will be included
    sections_info = ["✅ Portfolio Architecture & Risk Analytics", "✅ Tail Risk Analysis"]
    
    if data_dict['has_benchmark']:
        sections_info.append(f"✅ Benchmark Comparison ({data_dict['benchmark_name']})")
    
    if data_dict['has_optimization']:
        sections_info.append("✅ Optimization Analysis")
    
    if data_dict['has_frontier']:
        sections_info.append("✅ Efficient Frontier Position")
    
    sections_info.append("✅ Strategic Summary")
    
    with st.expander("📋Sections Analysed", expanded=False):
        for section in sections_info:
            st.write(section)
    
    if st.button("🚀 Generate Full Analysis", type="primary", use_container_width=True):
        analysis = generate_full_analysis(data_dict)
        st.session_state["ai_full_analysis"] = analysis
        st.session_state["ai_analysis_generated"] = True
        st.rerun()
    st.write("""
Note:
1. In case of ticker changes, regenerate the analysis to reflect updated data.
2. If API limits are hit, please wait for 1-2 minutes before regenerating.
""")

    if st.session_state.get("ai_analysis_generated", False):
        st.divider()
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            if st.button("🔄 Regenerate"):
                generate_full_analysis.clear()
                with st.spinner("Regenerating complete analysis..."):
                    analysis = generate_full_analysis(data_dict)
                    st.session_state["ai_full_analysis"] = analysis
                    st.rerun()
        
        with col2:
            if st.button("📊 Read Aloud"):
                with st.spinner("Generating audio..."):
                    b64 = generate_tts_audio(st.session_state["ai_full_analysis"])
                    if b64:
                        audio_html = f"""
                        <audio id="analysisAudio" controls autoplay controlsList="nodownload">
                            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                        </audio>
                        <script>
                        document.getElementById("analysisAudio").play();
                        </script>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
                        st.success("🎙️ Playing analysis!")
        
        st.divider()
        st.markdown(st.session_state["ai_full_analysis"])
    else:
        st.info("👆 Click 'Generate Full Analysis' for your comprehensive AI review.")