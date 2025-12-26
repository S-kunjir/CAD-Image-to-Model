from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
import traceback

from .chat_models import MODEL_TYPE, ChatModelParameters

_INSTRUCTIONS = """You are an expert Python/CAD debugging agent. Your task is to execute and fix CadQuery Python code.

## IMPORTANT RULES:
1. **Execute the code FIRST** to see what happens
2. **Make MINIMAL changes** to fix errors
3. **Preserve the original intent** - don't rewrite from scratch
4. **Check cadquery imports** - make sure cadquery is imported as `cq`
5. **Verify export statement** - ensure `cq.exporters.export(result, output_filename)` exists
6. **Handle common CadQuery errors**:
   - Invalid workplane selections
   - Missing imports
   - Incorrect method chaining
   - Invalid geometry operations

## WORKFLOW:
1. Run the code and observe errors
2. Fix ONE issue at a time
3. Re-run to verify fix
4. Repeat until code runs successfully

## IF CODE ALREADY WORKS:
Just run it and confirm success.

## IF UNFIXABLE:
If you cannot fix it after reasonable attempts, return "I cannot fix it."

Remember: CadQuery is already installed. Use `import cadquery as cq`.
Always keep the output file path variable as `output_filename`.
"""


def execute_python_code(code: str, model_type: MODEL_TYPE = "gpt", only_execute: bool = False) -> str:
    """
    Execute the given Python `code`. If it fails, attempt to fix and re-run.
    Returns the final agent output message.
    """
    tools = [PythonREPLTool()]

    if only_execute:
        try:
            return tools[0].run(code)
        except Exception as e:
            return f"Execution error: {str(e)}\n\n{traceback.format_exc()}"

    llm = ChatModelParameters.from_model_name(model_type).create_chat_model()

    agent = create_tool_calling_agent(llm, tools, ChatPromptTemplate.from_messages([
        ("system", _INSTRUCTIONS),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]))

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        early_stopping_method="generate",
    )

    user_input = (
        "Please execute this CadQuery Python code. If it doesn't work, fix the errors.\n"
        f"```python\n{code}\n```\n\n"
        "IMPORTANT: Keep the `output_filename` variable for file export."
    )

    try:
        result = agent_executor.invoke({"input": user_input})
        return result["output"]
    except Exception as e:
        return f"Agent execution error: {str(e)}\n\n{traceback.format_exc()}"