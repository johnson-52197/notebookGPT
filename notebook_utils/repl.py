import re
from typing import Dict, Tuple
from langchain import LLMChain, PromptTemplate
from langchain.schema import BaseLanguageModel
from pydantic import Field

from notebook_utils.utils import create_new_cell

repl_template = """You are an assistant on a jupter notebook using pandas, matplotlib and pyvis.
Your task is to generate a new cell based on the user's request.
Your response should be the cell body, nothing else. Do not surround it in snippet quotes or anything else.
The first thing in the cell will be a triple quoted string with a comment.

The user has requested the following:
{input}
New cell:
"""

repl_prompt = PromptTemplate(
        template=repl_template,
        input_variables=["input"],
        )


class REPLChain(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: PromptTemplate = repl_prompt, verbose=False) -> "REPLChain":
        return cls(llm=llm, prompt=prompt, verbose=verbose)
    
    def run(self, input: str, **kwargs):
        return create_new_cell(self({**kwargs, "input": input})[self.output_key])
