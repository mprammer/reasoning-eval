"""
This file contains the template for the reasoning task.
The template is designed to guide the model in generating a structured and clear response to math/code problems.
It includes instructions for formatting the answer and ensuring clarity in the explanation.

By CMU team
"""


MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

NL_TO_SQL_QUERY_TEMPLATE = """
Use the following SQL database schema and natural language query to create an SQL query. The last line of your response should be the SQL query. After your answer, provide a confidence score between 1 and 10 for your answer.

{Question}
""".strip()
