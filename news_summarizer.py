"""
News Summarization and Reformatting Module

This module provides functionality for summarizing and reformatting news content 
using the langchain framework and DeepSeek's R1 API service.
"""

import os
import json
import requests
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Any, Dict, List, Optional

# Load environment variables (for API keys)
load_dotenv()


class DeepSeekLLM(LLM):
    """LangChain integration for DeepSeek API"""
    
    api_key: str
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_base_url: str = "https://api.deepseek.com/v1/chat/completions"
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            data["stop"] = stop
            
        response = requests.post(
            self.api_base_url,
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(f"DeepSeek API call failed: {response.text}")
            
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


def create_prompt_templates():
    """Create the three prompt templates for different aspects of news filtering"""
    
    # Prompt 1: Filter AI-related news in Chinese stock market
    ai_china_stock_template = PromptTemplate(
        input_variables=["news_content"],
        template="""
任务描述：
请从以下新闻列表中挑选出15条与AI最相关最相关最重要的新闻。

要求：
1. 最多选择15条新闻（若符合条件的新闻不足15条，则输出实际条数）。
2. 筛选过程必须基于新闻事实，不允许编造信息。
3. 输出结果请使用中文，并按本身的顺序排序。
4. 输出每条新闻的原文，不需要添加额外说明。

数据源：
{news_content}

示例输出格式：
新闻标题1
新闻详情1
新闻标题2
新闻详情2
"""
    )

    # Prompt 2: Filter international economic news (excluding China)
    international_economic_template = PromptTemplate(
        input_variables=["news_content"],
        template="""
任务描述：
请从以下新闻列表中筛选出与国际经济（不包括中国）最相关、最重要的新闻。

要求：
1. 最多选择15条新闻（若符合条件的新闻不足15条，则输出实际条数）。
2. 筛选过程必须基于新闻事实，不允许编造信息。
3. 输出结果请使用中文，并按本身的顺序排序。
4. 输出每条新闻的原文，不需要添加额外说明。

数据源：
{news_content}

示例输出格式：
新闻标题1
新闻详情1
新闻标题2
新闻详情2
"""
    )

    # Prompt 3: Filter news relevant to Chinese stock market
    china_stock_template = PromptTemplate(
        input_variables=["news_content"],
        template="""
任务描述：
从以下新闻列表中挑选出15条与中国股市最相关最重要的新闻,最相关、最重要的新闻：

要求：
1. 最多选择15条新闻（若符合条件的新闻不足15条，则输出实际条数）。
2. 筛选过程必须基于新闻事实，不允许编造信息。
3. 输出结果请使用中文，并按本身的顺序排序。
4. 输出每条新闻的原文，不需要添加额外说明。
5. 去掉【】，去掉最开始的时间戳

数据源：
{news_content}

示例输出格式：
1. 新闻标题1
   新闻详情1
2. 新闻标题2
   新闻详情2
"""
    )
    
    return ai_china_stock_template, international_economic_template, china_stock_template


def create_llm_chains(api_key, temperature=0.7, max_tokens=1024):
    """Create the LLM and the three chains for processing news content"""
    
    # Initialize DeepSeek LLM
    llm = DeepSeekLLM(
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Get prompt templates
    ai_china_stock_template, international_economic_template, china_stock_template = create_prompt_templates()
    
    # Create three different processing chains
    ai_china_stock_chain = LLMChain(llm=llm, prompt=ai_china_stock_template)
    international_economic_chain = LLMChain(llm=llm, prompt=international_economic_template)
    china_stock_chain = LLMChain(llm=llm, prompt=china_stock_template)
    
    return llm, ai_china_stock_chain, international_economic_chain, china_stock_chain


def process_news_file(file_path, ai_china_stock_chain, international_economic_chain, china_stock_chain, verbose=True):
    """Process news file and generate filtered news by categories
    
    Args:
        file_path (str): Path to the news file
        ai_china_stock_chain (LLMChain): Chain for filtering AI-related Chinese stock market news
        international_economic_chain (LLMChain): Chain for filtering international economic news
        china_stock_chain (LLMChain): Chain for filtering Chinese stock market news
        verbose (bool): Whether to print progress messages
        
    Returns:
        str: Combined filtered news text
    """
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        news_content = f.read()
    
    if verbose:
        print(f"Processing file: {file_path}")
        print(f"File length: {len(news_content)} characters")
    
    # Run three different filtering chains
    if verbose:
        print("1. Filtering AI-related Chinese stock market news...")
    ai_china_stock_result = ai_china_stock_chain.run(news_content=news_content)
    
    if verbose:
        print("2. Filtering international economic news...")
    international_economic_result = international_economic_chain.run(news_content=news_content)
    
    if verbose:
        print("3. Filtering Chinese stock market news...")
    china_stock_result = china_stock_chain.run(news_content=news_content)
    
    # Combine results
    combined_summary = f"""# 新闻分类筛选结果

## AI相关中国股市新闻
{ai_china_stock_result}

## 国际经济新闻
{international_economic_result}

## 中国股市新闻
{china_stock_result}
"""
    
    return combined_summary


def save_summary(summary, input_file_path):
    """Save filtered news results to file
    
    Args:
        summary (str): The combined filtered news text
        input_file_path (str): The path to the original input file
        
    Returns:
        str: Path to the output file
    """
    # Generate output file path from input file path
    base_name = os.path.basename(input_file_path)
    file_name, _ = os.path.splitext(base_name)
    output_file = f"{file_name}_filtered_news.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Filtered news saved to: {output_file}")
    return output_file


def filter_news(file_path, api_key=None, temperature=0.7, max_tokens=1024):
    """Main function to filter news from a file into different categories
    
    Args:
        file_path (str): Path to the news file
        api_key (str, optional): DeepSeek API key. If None, will try to get from environment.
        temperature (float, optional): LLM temperature parameter. Defaults to 0.7.
        max_tokens (int, optional): Maximum tokens for LLM response. Defaults to 1024.
        
    Returns:
        tuple: (filtered_news_text, output_file_path)
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not found. Please provide an API key or set DEEPSEEK_API_KEY environment variable.")
    
    # Create LLM and chains
    _, ai_china_stock_chain, international_economic_chain, china_stock_chain = create_llm_chains(
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Process the news file
    summary = process_news_file(file_path, ai_china_stock_chain, international_economic_chain, china_stock_chain)
    
    # Save the summary
    output_file = save_summary(summary, file_path)
    
    return summary, output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter news by categories using DeepSeek API")
    parser.add_argument("file_path", help="Path to the news file to filter")
    parser.add_argument("--api_key", help="DeepSeek API key (if not set in environment)")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens for LLM response (default: 1024)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} does not exist.")
        exit(1)
    
    try:
        filtered_news, output_file = filter_news(
            args.file_path,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"\nNews successfully filtered and saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
