# %% [markdown]
# 여기서는 LLM이 자신이 하고 있는 일을 잘 하고 있는지 셀프로 검증하는 self-rag를 학습한다.

# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = Chroma(
    embedding_function=embeddings,
    collection_name='income_tax_collection',
    persist_directory='./income_tax_collection'
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})


# %%
# START -> RETRIEVE -> GENERATE -> END

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

graph_builder = StateGraph(AgentState)

# %%
def retrieve(state: AgentState):
    query = state['query']
    docs = retriever.invoke(query)
    return {'context': docs}    

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')

# %%
from langchain import hub

generate_prompt = hub.pull('rlm/rag-prompt')
generate_llm = ChatOpenAI(model='gpt-4o', max_tokens=100)
def generate(state: AgentState) -> AgentState:
    context = state['context']
    query = state['query']
    rag_chain = generate_prompt | llm
    response = rag_chain.invoke({'question': query, 'context': context})
    return {'answer': response.content}

# %%
from langchain import hub
from typing import Literal

# 질문과 document의 연관성이 얼마나 되는지 확인하는 rag-document-relevance 프롬프트
doc_relevance_prompt = hub.pull('langchain-ai/rag-document-relevance')

# self-rag에서는 END를 호출을 해야 하는데, END는 노드가 아니기 떄문에, Literal을 해도 에러가 발생한다. 따라서 엣지에서 처리해야 한다.
# 질문이 문서와 얼마나 연관성이 있는지 체크하는 노드
def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    context = state['context']
    query = state['query']
    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})
    if response['Score'] == 1:
        return 'relevant'
    return 'irrelevant'

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    사전: {dictionary}
    질문: {{query}}
"""
)

# 사용자의 질문을 변경해주는 노드
def rewrite(state: AgentState):
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    response = rewrite_chain.invoke({'query': query})
    return {'query': response}


# %%
from langchain import hub

hallucination_prompt = hub.pull('langchain-ai/rag-answer-hallucination')

# 할루시네이션이 발생하였는지를 체크하는 노드
def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    hallucination_chain = hallucination_prompt | llm
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})
    if response['Score'] == 1:
        return 'hallucinated'
    return 'not hallucinated'


# %%
from langchain import hub

helpfulness_prompt = hub.pull('langchain-ai/rag-answer-helpfulness')

# 질문과 답변이 연관이 있는지 체크하는 노드
def check_helpfulness_grader(state: AgentState) -> Literal['helpful', 'unhelpful']:
    query = state['query']
    answer = state['answer']
    helpfulness_chain = helpfulness_prompt | llm
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})
    if response['Score'] == 1:
        return 'helpful'
    return 'unhelpful'


def check_helpfulness(state: AgentState):
    return state

# %%
# 할루시네이션을 검증하는 prompt의 테스트

query = '연봉 5천만원인 거주자의 소득세는 얼마인가요?'

context = retriever.invoke(query)
generate_state = {'query': query, 'context': context}
answer = generate(generate_state)

# langchain 할루시네이션 프롬프트를 사용 시 할루시네이션이 발생했다고 판단한다.
# 문서에 있는 내용을 근거로 해서 답변을 해서 답변이 길어서로 유추된다.
# langchain에서 제공하는 prompt를 사용하지 않고, 직접 prompt를 수정해서 사용해야 할것 같다.
hallucination_state = {'answer': answer, 'context': context}

check_hallucination(hallucination_state)

# %%
# 할루시네이션을 검증하는 prompt의 테스트

query = '연봉 5천만원인 거주자의 소득세는 얼마인가요?'

context = retriever.invoke(query)
generate_state = {'query': query, 'context': context}
answer = generate(generate_state)

helpfulness_state = {'query': query, 'answer': answer}

check_helpfulness(helpfulness_state)

# %%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)


# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'hallucinated': 'generate',
        'not hallucinated': 'check_helpfulness'
    }
)
graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite', 'retrieve')

# %%
graph = graph_builder.compile()



