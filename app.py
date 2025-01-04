import streamlit as st
import json
import re
from typing import Optional, Dict, List
from statistics import mean, median
from openai import AsyncOpenAI
import anthropic
from anthropic import AsyncAnthropic
import matplotlib.pyplot as plt
import numpy as np
import toml
from pathlib import Path
import asyncio
import nest_asyncio

# Enable nested asyncio for Streamlit compatibility
nest_asyncio.apply()

# Pricing per 1000 tokens (as of December 2024)
MODEL_PRICING = {
    # OpenAI Models
    "gpt-4o": {"input": 0.01, "output": 0.03},
    "o1-preview": {"input": 0.01, "output": 0.03},
    "o1-mini": {"input": 0.001, "output": 0.002},
    # Anthropic Models
    "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015}
}

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "gpt-4o": {"samples": 10, "weight": 1},
    "o1-preview": {"samples": 1, "weight": 1},
    "o1-mini": {"samples": 3, "weight": 1},
    "claude-3-5-sonnet-latest": {"samples": 10, "weight": 1}
}

class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.model_usage: Dict[str, Dict[str, int]] = {}

    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        if model not in self.model_usage:
            self.model_usage[model] = {"input_tokens": 0, "output_tokens": 0}

        self.model_usage[model]["input_tokens"] += input_tokens
        self.model_usage[model]["output_tokens"] += output_tokens

        if model in MODEL_PRICING:
            input_cost = (input_tokens / 1000) * MODEL_PRICING[model]["input"]
            output_cost = (output_tokens / 1000) * MODEL_PRICING[model]["output"]
            self.total_cost += input_cost + output_cost

    def get_summary(self) -> str:
        return f"Total API cost: ${self.total_cost:.4f}"

async def process_openai_query_async(query: str, model: str, openai_client, cost_tracker) -> Optional[float]:
    """Process a query using OpenAI's API asynchronously"""
    try:
        if model in ["o1-mini", "o1-preview"]:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": f"{st.session_state.general_prompt}\n{query}"}
                ]
            )
        else:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": st.session_state.general_prompt},
                    {"role": "user", "content": query}
                ]
            )

        cost_tracker.add_usage(
            model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )

        response_text = response.choices[0].message.content
        return extract_number(response_text, model)
    except Exception as e:
        st.error(f"Error with OpenAI model {model}: {str(e)}")
        return None

async def process_anthropic_query_async(query: str, model: str, anthropic_client, cost_tracker) -> Optional[float]:
    """Process a query using Anthropic's API asynchronously"""
    try:
        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            system=st.session_state.general_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )

        cost_tracker.add_usage(
            model,
            response.usage.input_tokens,
            response.usage.output_tokens
        )

        response_text = response.content[0].text
        return extract_number(response_text, model)
    except Exception as e:
        st.error(f"Error with Anthropic model {model}: {str(e)}")
        return None

def extract_number(response_text: str, model: str) -> Optional[float]:
    """Extract numerical answer from the response"""
    match = re.search(r"<answer>(.*?)</answer>", response_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            st.error(f"Error: Could not convert answer to float for model {model}")
            return None
    else:
        st.error(f"Error: No <answer> tag found in the response from model {model}")
        return None

def calculate_weighted_average(model_averages: Dict[str, float], config: Dict) -> Optional[float]:
    """Calculate weighted average of results using pre-averaged model results"""
    if not model_averages:
        return None

    total_weighted_sum = 0.0
    total_weight = 0.0

    for model, avg_value in model_averages.items():
        weight = config[model]['weight']
        total_weighted_sum += avg_value * weight
        total_weight += weight

    return total_weighted_sum / total_weight if total_weight > 0 else None

def format_number(num: float) -> str:
    """Format number for display with appropriate units and precision"""
    abs_num = abs(num)
    if abs_num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{num/1_000:.1f}K"
    elif abs_num >= 100:
        return f"{num:.1f}"
    elif abs_num >= 1:
        return f"{num:.2f}"
    else:
        return f"{num:.3f}"

def visualize_results(model_results: Dict[str, List[float]], model_averages: Dict[str, float], weighted_avg: float):
    """Create visualization of model results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Box plot
    data_to_plot = []
    labels = []
    for model, results in model_results.items():
        if results:
            data_to_plot.append(results)
            labels.append(model.split('-')[0])

    bp = ax1.boxplot(data_to_plot, patch_artist=True)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_title('Distribution of Model Responses')
    ax1.set_ylabel('Predicted Value')

    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Plot 2: Bar plot
    models = list(model_averages.keys())
    averages = list(model_averages.values())
    x = np.arange(len(models))

    bars = ax2.bar(x, averages, alpha=0.7, color='lightblue')
    ax2.axhline(y=weighted_avg, color='red', linestyle='--', label='Weighted Average')

    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split('-')[0] for m in models], rotation=45)
    ax2.set_title('Model Averages vs Weighted Average')
    ax2.set_ylabel('Value')
    ax2.legend()

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')

    plt.tight_layout()
    return fig

def initialize_clients():
    """Initialize API clients using Streamlit secrets"""
    try:
        openai_client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        anthropic_client = AsyncAnthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        return openai_client, anthropic_client
    except Exception as e:
        st.error(f"Error initializing API clients: {str(e)}")
        st.info("Please ensure you have set up your .streamlit/secrets.toml file with OPENAI_API_KEY and ANTHROPIC_API_KEY")
        return None, None

async def process_model_queries(model: str, settings: dict, query: str, openai_client, anthropic_client, cost_tracker) -> List[float]:
    """Process all queries for a single model in parallel"""
    tasks = []
    for _ in range(settings["samples"]):
        if model in ["gpt-4o", "o1-preview", "o1-mini"]:
            task = process_openai_query_async(query, model, openai_client, cost_tracker)
        else:
            task = process_anthropic_query_async(query, model, anthropic_client, cost_tracker)
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def main():
    st.set_page_config(page_title="AI Model Comparison", layout="wide")
    st.title("AI Model Comparison Dashboard")

    # Initialize session state
    if 'general_prompt' not in st.session_state:
        st.session_state.general_prompt = """
        You are an intelligent assistant. Respond to the user's query with a single numerical answer.
        The response should be inside an XML tag <answer>. For example: <answer>62.0</answer>
        """
    if 'query' not in st.session_state:
        st.session_state.query = "Estimate the median age at first birth of mothers worldwide"

    # Initialize API clients
    openai_client, anthropic_client = initialize_clients()
    if not openai_client or not anthropic_client:
        return

    # Sidebar for configuration
    st.sidebar.header("Model Configuration")

    # Model configuration
    config = {}
    for model, default_config in DEFAULT_MODEL_CONFIG.items():
        st.sidebar.subheader(f"{model} Configuration")
        samples = st.sidebar.number_input(f"Number of samples for {model}",
                                        min_value=0,
                                        max_value=default_config["samples"],
                                        value=default_config["samples"],
                                        key=f"{model}_samples")
        weight = st.sidebar.number_input(f"Weight for {model}",
                                       min_value=0.0,
                                       max_value=10.0,
                                       value=float(default_config["weight"]),
                                       key=f"{model}_weight")
        config[model] = {"samples": samples, "weight": weight}

    # Query input
    query = st.text_input("Enter your query:",
                         value=st.session_state.query,
                         key="query_input",
                         on_change=lambda: setattr(st.session_state, 'query', st.session_state.query_input))

    # Process button
    if st.button("Process Query"):
        with st.spinner("Processing query across models..."):
            cost_tracker = CostTracker()
            model_averages = {}
            all_results = {}

            # Create columns for progress bars
            cols = st.columns(len(config))
            progress_bars = {model: cols[i].progress(0)
                           for i, model in enumerate(config.keys())}

            # Process all models in parallel
            async def process_all_models():
                tasks = []
                for model, settings in config.items():
                    if settings["samples"] > 0:
                        task = process_model_queries(model, settings, query, openai_client, anthropic_client, cost_tracker)
                        tasks.append((model, task))

                for model, task in tasks:
                    results = await task
                    if results:
                        all_results[model] = results
                        model_averages[model] = mean(results)
                    progress_bars[model].progress(1.0)

            # Run all queries in parallel
            asyncio.run(process_all_models())

            if all_results:
                st.subheader("Results")

                # Calculate statistics
                all_individual_results = [result for results in all_results.values()
                                        for result in results]
                weighted_avg = calculate_weighted_average(model_averages, config)
                simple_mean = mean(all_individual_results)
                med = median(all_individual_results)
                q1 = np.percentile(all_individual_results, 25)
                q3 = np.percentile(all_individual_results, 75)

                # Calculate mean without outliers (using IQR method)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_results = [x for x in all_individual_results if lower_bound <= x <= upper_bound]
                mean_no_outliers = mean(filtered_results) if filtered_results else simple_mean

                # Display key statistics with emphasis
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Median")
                    st.markdown(f"<h2 style='color: #1f77b4;'>{format_number(med)}</h2>", unsafe_allow_html=True)
                with col2:
                    st.markdown("### Simple Mean")
                    st.markdown(f"<h2 style='color: #2ca02c;'>{format_number(simple_mean)}</h2>", unsafe_allow_html=True)
                with col3:
                    st.markdown("### Weighted Average")
                    st.markdown(f"<h2 style='color: #ff7f0e;'>{format_number(weighted_avg)}</h2>", unsafe_allow_html=True)

                # Display additional statistics in a more subtle way
                st.markdown("#### Additional Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("First Quartile (Q1)", format_number(q1))
                with col2:
                    st.metric("Third Quartile (Q3)", format_number(q3))
                with col3:
                    st.metric("Mean (No Outliers)", format_number(mean_no_outliers))
                with col4:
                    st.metric("Minimum", format_number(min(all_individual_results)))
                with col5:
                    st.metric("Maximum", format_number(max(all_individual_results)))

                # Display visualization
                fig = visualize_results(all_results, model_averages, weighted_avg)
                st.pyplot(fig)

                # Display cost summary
                st.info(cost_tracker.get_summary())
            else:
                st.error("Failed to retrieve any numerical results.")

if __name__ == "__main__":
    main()
