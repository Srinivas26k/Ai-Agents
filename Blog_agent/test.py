import streamlit as st
from phi.agent import Agent
from phi.tools.googlesearch import GoogleSearch
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.crawl4ai_tools import Crawl4aiTools
from phi.model.groq import Groq
from textwrap import dedent
from datetime import datetime

groq_api_key = st.secrets["GROQ_API_KEY"]

# Define Researcher Agent
researcher_agent = Agent(
    name='Keyword_Competitor_Research_Agent',
    tools=[GoogleSearch(), DuckDuckGo(), Crawl4aiTools()],
    model = Groq(id="deepseek-r1-distill-llama-70b", api_key=groq_api_key),
    description="Finds high-ranking keywords and analyzes top competitor blogs to optimize SEO content.",
    instructions=[
        "Fetch trending and high-volume, low-competition keywords.",
        "Analyze top-ranking competitor blogs for keyword usage, meta tags, and structure.",
        "Extract semantic keywords (LSI terms) and suggest relevant topic clusters.",
        "Identify frequently asked questions (FAQs) from search engines for content optimization."
    ],
    show_tool_calls=True,
    markdown=True,
    expected_output=dedent("""\
        ## Keyword Research Report

        ### High-Ranking Keywords
        - **Primary Keyword:** {keyword}
        - **Search Volume:** {volume}
        - **Competition Level:** {low/medium/high}

        ### Competitor Analysis
        **Competitor:** {competitor_name}  
        **Ranking Position:** {position}  
        **Meta Description:** "{meta_description}"  
        **Backlink Sources:**  
        - [Source 1](link)  
        - [Source 2](link)  

        ### Semantic Keywords (LSI Terms)
        - {keyword_1}
        - {keyword_2}
        - {keyword_3}

        ### Suggested Topic Clusters
        - {topic_cluster_1}
        - {topic_cluster_2}
        - {topic_cluster_3}

        ### Frequently Asked Questions (FAQs)
        1. {question_1}
        2. {question_2}
        3. {question_3}

        ### Summary
        {brief report on findings}

        ### References
        - [Google Keyword Trends](link)
        - [Competitor Blog](link)
        - [Backlink Source](link)

        - Published on {date} in dd/mm/yyyy
        """)
)


# Define Content Writer Agent
content_writer_agent = Agent(
    name="AIContent_Writer_Agent",
    model = Groq(id="deepseek-r1-distill-llama-70b", api_key=groq_api_key),
    description="Generates high-quality, SEO-optimized content based on keyword research and competitor analysis.",
    instructions=[
        "Use keywords provided by the research agent to generate well-structured blog content.",
        "Ensure a natural flow in writing to avoid AI detection issues.",
        "Incorporate headings, subheadings, bullet points, and FAQs."
    ],
    show_tool_calls=True,
    markdown=True,
    expected_output=dedent("""\
        ## {Blog Title}

        ### Introduction
        {Write an engaging introduction that hooks the reader and introduces the topic.}
        
        ### Main Content Sections
        #### {Section 1 Title}
        {Detailed explanation of Section 1.}

        #### {Section 2 Title}
        {Detailed explanation of Section 2.}

        #### {FAQs}
        1. **{Question 1}?**  
           {Answer 1}
        2. **{Question 2}?**  
           {Answer 2}

        ### Conclusion
        {Summarize the article and provide a compelling closing statement.}

        ### SEO Optimization Summary
        - **Keyword Usage:** {percentage}%
        - **Readability Score:** {score}
        - **Meta Description:** "{meta_description}"

        ### References
        - [Source 1](link)
        - [Source 2](link)

        - Published on {date} in dd/mm/yyyy
        """)
)

# Define On-Page SEO Agent
optimiser_agent = Agent(
    name="OnPage_SEO_Agent",
    model = Groq(id="deepseek-r1-distill-llama-70b", api_key=groq_api_key),
    description="Optimizes blog structure, metadata, and readability for better search engine ranking.",
    instructions=[
        "Generate and insert SEO-friendly meta titles, descriptions, and URL slugs.",
        "Optimize heading structures (H1-H6) for better readability and keyword placement."
    ],
    show_tool_calls=True,
    markdown=True,
    expected_output=dedent("""\
        ## SEO Optimization Report

        ### Meta Data
        - **Meta Title:** {optimized_title}
        - **Meta Description:** "{optimized_description}"
        - **URL Slug:** {optimized_url}

        ### Heading Structure Optimization
        **Original Structure:**  
        - H1: {original_h1}  
        - H2: {original_h2}  

        **Optimized Structure:**  
        - H1: {optimized_h1}  
        - H2: {optimized_h2}  
        - H3: {optimized_h3}

        ### Keyword Density Analysis
        - **Primary Keyword:** {keyword}
        - **Density:** {percentage}%
        - **Long-Tail Keywords Used:** {yes/no}

        ### Readability Score
        - **Flesch Reading Ease:** {score}
        - **Passive Voice Percentage:** {percentage}%
        - **Sentence Length Optimization:** {yes/no}

        ### Image Optimization
        - **Alt Text Added:** {yes/no}
        - **File Size Optimization:** {yes/no}

        ### Internal & External Links Check
        - **Internal Links Added:** {yes/no}
        - **Broken Links Found:** {yes/no}

        ### Recommendations
        {list of final recommendations}

        ### References
        - [Google SEO Guide](link)
        - [SEO Tool Report](link)

        - Published on {date} in dd/mm/yyyy
        """)
)

# Define the Blog AI Agent (Team)
blog_ai_agent = Agent(
    team=[researcher_agent, content_writer_agent, optimiser_agent],
    instructions=["Always include sources, references, and citations for factual information."],
    show_tool_calls=True,
    markdown=True,
    model = Groq(id="deepseek-r1-distill-llama-70b", api_key=groq_api_key)
)

# Streamlit UI Setup
st.set_page_config(page_title="SEO AI Chatbot", page_icon="💬", layout="wide")
st.title("💬 AI-Powered SEO Chatbot")

st.title("AI Chat Assistant")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
prompt = st.chat_input("Ask me anything about SEO, blogging, and keyword research...")

if prompt:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in blog_ai_agent.run(prompt, stream=True):
                full_response += chunk.content
                response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {str(e)}"
            response_placeholder.markdown(full_response)

    # Save response in chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})